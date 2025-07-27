import argparse
from tempt_config import TEMPTConfig
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from TEMPT import TEMPT
from data_loader import *
from torchinfo import summary
from utils import EarlyStopping
from tqdm import tqdm
from pathlib import Path
import os
import yaml
from loss import *


def train(config: TEMPTConfig):
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    device = torch.device(config.device)
    model = TEMPT(max_len=config.seq_len, patch_sizes=config.patch_sizes, hidden_dims=config.hidden_dims, reduction='sum').to(device)
    
    # prepare data
    flag = "train"
    shuffle_flag = False
    train_dataset = TrajDataset(dataset=config.dataset, data_path_prefix='./data/', data_path=config.data_path, size=[config.seq_len, config.pred_len], flag=flag)
    print(flag, len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=shuffle_flag, num_workers=config.num_workers)

    val_dataset = TrajDataset(dataset=config.dataset, data_path_prefix='./data/', data_path=config.data_path, size=[config.seq_len, config.pred_len], flag="val")
    vali_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=shuffle_flag, num_workers=config.num_workers)


    summary(model)
    # prepare optimizer
    early_stopping = EarlyStopping(patience=config.patience,
                                       verbose=True)

    model_optim = torch.optim.Adam(model.parameters(),
                                       lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optim,
                                                            'min',
                                                            0.5,
                                                            1,
                                                            verbose=True)
    # loss = smape_loss()

    loss_func = nn.MSELoss()
    
    # prepare saving
    exp_dir = Path(config.save_dir) / config.task_name / timestamp

    os.makedirs(exp_dir, exist_ok=True)
    with open(exp_dir / "hparams.yaml", 'w',
                  encoding="utf8") as f:
            yaml.dump(config, f, yaml.Dumper)

    # train
    results = []

    with torch.cuda.device(device):
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(config.train_epochs):
            iter_count = 0
            train_loss = []

            model.train()
            for i, (batch_x, batch_y) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, mininterval=5)):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)

                predictions, residual = model(batch_x, None)

                pred_loss = loss_func(predictions, batch_y)
                res_loss = residual_loss_fn(residual, 0., .1, eps=1e-5)
                loss = pred_loss + res_loss
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)

            # val
            val_loss = []
            model.eval()
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                predictions, residual = model(batch_x, None)
                pred_loss = loss_func(predictions, batch_y)
                val_loss.append(pred_loss.item())
            val_loss = np.average(val_loss)

            results.append({
                'Epoch': epoch,
                'train_loss': train_loss,
                "val_loss": val_loss,
                "lr": model_optim.param_groups[0]['lr']
            })
            print(f"epoch: {epoch}, train_loss: {train_loss:.3f}, " +
                    f"val_loss: {val_loss:.3f}, " +
                    f"lr: {model_optim.param_groups[0]['lr']}")
            early_stopping(val_loss, model, str(exp_dir))
            if early_stopping.early_stop:
                break
            scheduler.step(val_loss)

    df = pd.DataFrame(results)
    df.to_csv(exp_dir / 'metrics.csv', index=False)
    # best_model_path = exp_dir / 'checkpoint.pth'
    # model.load_state_dict(torch.load(best_model_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "TEMPT Train: TrajEctory Mixing decomPosiTion Train"
    )

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default="XA")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patch_sizes", type=str, default="[40, 20, 10, 1]")
    parser.add_argument("--hidden_dims", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--pred-len", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default="logs")
    parser.add_argument("--mode", choices=["long", "short", "all"], default="short")

    args = parser.parse_args()

    config = TEMPTConfig()
    config.device = args.device
    config.dataset = args.dataset
    config.data_path = "{}term_in{}_out{}.npy".format(args.mode, args.seq_len, args.pred_len)
    config.batch_size = args.batch_size
    config.train_epochs = args.epochs
    config.patch_sizes = eval(args.patch_sizes)
    config.hidden_dims = [args.hidden_dims] * len(config.patch_sizes)
    config.seq_len = args.seq_len
    config.pred_len = args.pred_len
    config.save_dir = args.save_dir
    train(config)
