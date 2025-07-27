import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from timm.layers import DropPath
from loss import *
from torchinfo import summary

def get_activation(activ: str):
    if activ == "gelu":
        return nn.GELU()
    elif activ == "sigmoid":
        return nn.Sigmoid()
    elif activ == "tanh":
        return nn.Tanh()
    elif activ == "relu":
        return nn.ReLU()

    raise RuntimeError("activation should not be {}".format(activ))

class MLPBlock(nn.Module):

    def __init__(
        self,
        mix_dim,
        in_features: int,
        hid_features: int,
        out_features: int,
        activ="gelu",
        drop: float = 0.0,
        jump_conn='trunc',
    ):
        super().__init__()
        self.dim = mix_dim
        self.out_features = out_features
        self.net = nn.Sequential(
            nn.Linear(in_features, hid_features),
            get_activation(activ),
            nn.Linear(hid_features, out_features),
            DropPath(drop))
        if jump_conn == "trunc":
            self.jump_net = nn.Identity()
        elif jump_conn == 'proj':
            self.jump_net = nn.Linear(in_features, out_features)
        else:
            raise ValueError(f"jump_conn:{jump_conn}")

    def forward(self, x):
        x = torch.transpose(x, self.dim, -1)
        x = self.jump_net(x)[..., :self.out_features] + self.net(x)
        x = torch.transpose(x, self.dim, -1)
        return x

class PatchMixer(nn.Module):
    def __init__(self, patch_size, patch_hidden, channel_dim, hidden_dim, norm=None, active='gelu'):
        super(PatchMixer, self).__init__()
        self.patch_size = patch_size
        self.channel_dim = channel_dim

        # Mixer for patch_size dimension
        self.mlp_patch = MLPBlock(mix_dim=2, in_features=patch_size, hid_features=patch_hidden, out_features=patch_size, activ=active)

        # Mixer for channel dimension
        self.mlp_channel = MLPBlock(mix_dim=3, in_features=channel_dim, hid_features=hidden_dim, out_features=channel_dim, activ=active)
        if norm == 'bn':
            norm_class = nn.BatchNorm2d
        elif norm == 'in':
            norm_class = nn.InstanceNorm2d
        else:
            norm_class = nn.Identity
        self.norm_class = norm_class(channel_dim)

    def forward(self, x):
        # x: [batch_size, seq_length//patch_size, patch_size, channel]
        x = self.norm_class(x)
        x = self.mlp_channel(x)
        x = self.norm_class(x)
        x = self.mlp_patch(x)
        return x
        # Transpose patch and channel for patch mixer
    
class DecompositionHead(nn.Module):
    def __init__(self, patch_size, hidden_len, seq_len, in_channel, hidden_dim, out_channel, active='gelu'):
        super(DecompositionHead, self).__init__()
        self.patch_size = patch_size
        # self.channel_dim = channel_dim

        # Mixer for patch_size dimension
        self.mlp_patch = MLPBlock(mix_dim=1, in_features=patch_size, hid_features=hidden_len, out_features=seq_len, activ=active, jump_conn='proj')

        # Mixer for channel dimension
        self.mlp_channel = MLPBlock(mix_dim=2, in_features=in_channel, hid_features=hidden_dim, out_features=out_channel, activ=active)

    def forward(self, x):
        # x: [batch_size, patch_size, channel]
        x = self.mlp_channel(self.mlp_patch(x))

        return x
    
class WeightedPooling(nn.Module):
    def __init__(self, num_patches):
        super(WeightedPooling, self).__init__()
        self.weights = nn.Parameter(torch.linspace(0.1, 1.0, num_patches))  # Learnable weights

    def forward(self, x, mask=None):
        # x: [batch_size, num_patches, patch_size, channel]
        weights = self.weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, num_patches, 1, 1]
        if mask is not None:
            weights = weights * mask.unsqueeze(-1).unsqueeze(-1)  # Apply mask if provided
        weighted_sum = (x * weights).sum(dim=1)  # Weighted sum over patches
        return weighted_sum
    
class PredictionHead(nn.Module):
    def __init__(self, patch_size, pred_len, hidden_len, in_channel, hidden_dim, out_channel, active='gelu'):
        super(PredictionHead, self).__init__()
        self.channel_mlp = MLPBlock(mix_dim=2, in_features=in_channel, hid_features=hidden_dim, out_features=out_channel, activ=active)
        self.patch_mlp = MLPBlock(mix_dim=1, in_features=patch_size, hid_features=hidden_len, out_features=pred_len, activ=active, jump_conn='proj')

        self.cache_mlp = MLPBlock(mix_dim=2, in_features=in_channel, hid_features=hidden_dim, out_features=out_channel, activ=active)  # Inference cache for self-autoregressive prediction

        self.final_mixer = MLPBlock(mix_dim=2, in_features=out_channel*2, hid_features=hidden_dim, out_features=out_channel, activ=active)

    def step(self, input_data, inference_cache=None):
        """
        Single-step prediction.
        If cache is None, use the given input_data.
        Otherwise, use the cached value for prediction.
        """
        mixed_input = self.channel_mlp(self.patch_mlp(input_data)) # [batch_size, 1, channel]
        if inference_cache is not None:
            # If cache exists, process both input and cache
            mixed_cache = self.cache_mlp(inference_cache)  # [batch_size, 1, channel]

            # Combine and process through final mixer
            combined = torch.cat([mixed_input, mixed_cache], dim=-1)  # [batch_size, 1, 2 * channel]
            output = self.final_mixer(combined)  # [batch_size, 1, channel]
            return output
        else:
            # print(mixed_input.shape)
            return mixed_input
    
    def forward(self, x, prediction_steps=10, inference_cache=None):
        mixed_input = self.channel_mlp(self.patch_mlp(x)) # [batch_size, pred_len, channel]
        return mixed_input

class TEMPT(nn.Module):
    def __init__(self, max_len=400, pred_len=10, patch_sizes=[10, 5], hidden_dims=[32, 16], reduction="sum"):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.hidden_dims = hidden_dims
        self.reduction = reduction
        self.max_len = max_len

        self.patch_mixers = nn.ModuleList()
        self.patch_poolings = nn.ModuleList()
        self.decomposition_heads = nn.ModuleList()
        self.pred_heads = nn.ModuleList()

        self.paddings = []

        for i, patch_size in enumerate(patch_sizes):
            patch_num = max_len // patch_size
            self.patch_mixers.append(PatchMixer(patch_size=patch_size, patch_hidden=16, channel_dim=2, hidden_dim=hidden_dims[i]))
            self.patch_poolings.append(WeightedPooling(num_patches=patch_num))
            self.decomposition_heads.append(DecompositionHead(patch_size=patch_size, hidden_len=128, seq_len=400, in_channel=2, hidden_dim=32, out_channel=2))
            self.pred_heads.append(PredictionHead(patch_size=patch_size, pred_len=pred_len, hidden_len=128, in_channel=2, hidden_dim=32, out_channel=2))

    
    def forward(self, x, input_mask):
        # x: [batch_size, seq_length, channel]
        batch_size, seq_length, channel = x.size()
        # assert seq_length % self.patch_size == 0, "seq_length must be divisible by patch_size"
        y_pred = []
        for i in range(len(self.patch_sizes)):           
            x_in = x
            num_patches = seq_length // self.patch_sizes[i]
            # Reshape input into patches
            x_in = x_in.view(batch_size, num_patches, self.patch_sizes[i], channel)  # [batch_size, num_patches, patch_size, channel]
            x_in = self.patch_mixers[i](x_in)
            # Apply weighted pooling
            pooled_output = self.patch_poolings[i](x_in, input_mask)  # [batch_size, patch_size, channel]
            comp = self.decomposition_heads[i](pooled_output)
            pred = self.pred_heads[i](pooled_output)
            y_pred.append(pred)
            x = x - comp

        y_pred = reduce(torch.stack(y_pred, 0), "h b l c -> b l c",
                        self.reduction)
        return y_pred, x
    
if __name__ == "__main__":
    model = TEMPT(max_len=400, patch_sizes=[40, 20, 10, 1], hidden_dims=[16, 16, 16, 16], reduction='sum')
    summary(model)
    