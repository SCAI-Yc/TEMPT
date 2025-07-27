class TEMPTConfig():
    def __init__(self) -> None:
        self.task_name="TEMPT_longterm_ar_prediction"
        self.save_dir="logs"
        self.use_gpu=True
        self.device="cuda:0" if self.use_gpu else "cpu"
        self.data="m3"
        self.dataset="CD"
        self.data_path="./data/XA_longterm_in400_out10.npy"
        self.num_workers=4
        self.in_chn=2
        self.out_chn=2
        self.itr=1
        self.batch_size=16
        self.patience=3
        self.learning_rate=1e-4
        self.train_epochs=30
        
        self.reduction = 'sum'
        self.norm = None
        self.last_norm = True
        self.drop = 0
        self.activ = "gelu"

        self.patch_sizes = [10, 5, 1, 1]
        self.hid_len = 64
        self.hid_chn = 32
        self.hid_pch = 32
        self.hid_pred = 64

        self.hidden_dims = [32, 32, 32, 32]

        self.seq_len = 10
        self.pred_len = 10

        self.data_norm_config = {
            'CD': {'Longitude': {'mean': 104.07658009, 'std': 0.0207446}, 'Latitude': {'mean': 30.68239038, 'std': 0.01855218}},
            'XA': {'Longitude': {'mean': 108.9467292, 'std': 0.02251446}, 'Latitude': {'mean': 34.24540809, 'std': 0.01980782}},
            }