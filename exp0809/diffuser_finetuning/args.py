class FinetuneArgs:
    def __init__(self):
        # My Configs
        self.bucket_name = "leo_font"
        self.save_path = "exp0809/finetune"
        self.content_path = "/home/jupyter/ai_font/data/exp0717/train0730_whole"
        self.style_path = "/home/jupyter/ai_font/data/exp0717/train0730_whole"
        self.target_path = "/home/jupyter/ai_font/data/exp0717/train0730_whole"
        self.finetune_style_path = ""
        self.finetune_target_path = ""
        self.scr = False
        self.num_neg = None
        self.experiment_name = "finetune"
        self.resolution=96
        # self.content_font = '시스템굴림'
        
        self.train_batch_size=16
        self.max_train_steps=100000
        self.save_freq = 10

        
        # Given
        self.unet_channels=(64, 128, 256, 512,)
        self.beta_scheduler="scaled_linear"
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_weight_decay = 1e-2
        self.adam_epsilon = 1e-08
        self.max_grad_norm = 1.0
        self.seed = 123
        self.style_image_size=self.resolution
        self.content_image_size=self.resolution
        self.content_encoder_downsample_size=3 
        self.channel_attn=True 
        self.content_start_channel=64 
        self.style_start_channel=64 

        self.perceptual_coefficient=0.01 
        self.offset_coefficient=0.5 
        self.ckpt_interval=40000 
        self.gradient_accumulation_steps=1 
        self.log_interval=50 
        self.learning_rate=1e-4 
        self.lr_scheduler="linear" 
        self.lr_warmup_steps=100 
        self.drop_prob=0.1 
        self.mixed_precision="no"
