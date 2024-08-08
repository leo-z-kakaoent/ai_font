import argparse
import os
from utils import *
import torch
# import models
# import data
import string

class BaseOptions():
    def __init__(self):
        self.dataroot = None
        self.ttfRoot = None
        self.corpusRoot = None
        self.name = "experiment_name"
        self.gpu_ids = "0"
        self.checkpoints_dir = "./checkpoints"
        self.model = "text_gan"
        self.init_loss = "Perceptual"
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 16
        self.D_ch = 16
        self.G_ch = 64
        self.netG = "resnet_6blocks"
        self.norm = "instance"
        self.init_type = "normal"
        self.init_gain = 0.02
        self.no_dropout = False
        self.num_writer = 1000
        self.num_writer_emb = 16
        # self.num_threads = 4
        # self.batch_size = 1
        self.imgH = 128
        self.imgW = 128
        self.display_winsize = 200
        self.epoch = "latest"
        self.load_iter = 0
        self.verbose = False
        self.suffix = ""
        
        self.resolution = 128
        self.batch_size = 32
        self.num_threads = 8
        self.datapath = "/home/jupyter/ai_font/data"
        self.savepath = "exp0604/cggan"
        self.experiment_name = "cggan_INfix"
        self.isTrain = True
        
        self.n_alphabet = 11172
        self.bucket_name = "leo_font"        
        

class TrainOptions(BaseOptions):

    def __init__(self):
        super().__init__()
        
        self.display_freq = 160
        self.display_ncols = 4
        self.display_id = 0
        self.display_server = "http://localhost"
        self.display_env = "main"
        self.display_port = 8097
        self.update_html_freq = 1000
        self.print_freq = 160
        self.no_html = False
        self.save_latest_freq = 4000
        self.save_epoch_freq = 1
        self.save_by_iter = False
        self.continue_train = False
        self.epoch_count = 1
        self.phase = "train"
        self.niter = 0
        self.niter_decay = 100
        self.beta1 = 0.5
        self.lr = 0.0001
        self.lrG = 0.0001
        self.lrD = 0.0001
        self.gan_mode = "lsgan"
        self.lr_policy = "linear"
        self.lr_decay_iters = 50
        self.hidden_size = 256
        self.dropout_p = 0.1
        self.max_length = 64
        self.val_num = 10
        self.dictionaryRoot = None  # Ensure this is set by the user
        self.val_seenstyleRoot = None  # Ensure this is set by the user
        self.val_unseenstyleRoot = None  # Ensure this is set by the user

        
class SamplingOptions(TrainOptions):
    
    def __init__(self, itern="0"):
        super().__init__()
        self.tag = "cggan"
        self.savefd = f"/home/jupyter/ai_font/data/reports/exp0604/cggan/i{itern}"
        self.bucket_name = "leo_font"
        self.content_encoder_path = f"exp0604/cggan/cggan_INfix__netContentEncoder__{itern}.pth.pth"
        self.style_encoder_path = f"exp0604/cggan/cggan_INfix__netStyleEncoder__{itern}.pth.pth"
        self.decoder_path = f"exp0604/cggan/cggan_INfix__netdecoder__{itern}.pth.pth"