class DefaultConfig:
    def __init__(self):
        self.lr = 0.0001
        self.dataset_path = "../data/raw_128/"
        self.batch_size = 64
        self.G_ch = 64
        self.n_embedding = 1024

class SCRConfig(DefaultConfig):
    def __init__(self):
        super().__init__()
        self.scr_dataset_path = "../data/raw_96/"
        self.scr_model_path = "../data/model/scr_98.pth"