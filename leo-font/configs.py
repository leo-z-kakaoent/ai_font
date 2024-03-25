import os

class DefaultConfig:
    def __init__(self):
        self.curdir = os.path.dirname(os.path.abspath(__file__))
        self.main_path = self.curdir.replace("/leo-font", "")
        self.lr = 0.0001
        self.dataset_path = f"{self.main_path}/data/raw_png_128/"
        self.batch_size = 64
        self.G_ch = 64
        self.n_embedding = 1024
        self.n_epoch = 1000

class SCRConfig(DefaultConfig):
    def __init__(self):
        super().__init__()
        self.scr_dataset_path = f"{self.main_path}/data/raw_png_96/"
        self.scr_model_path = f"{self.main_path}/data/model/scr_98.pth"
        self.scr_coef = 0.01