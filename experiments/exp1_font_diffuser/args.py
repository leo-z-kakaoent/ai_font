class TrainPhase1Args:
    def __init__(self):
        
        self.unet_channels=(64, 128, 256, 512,)
        self.beta_scheduler="scaled_linear"
        self.path = "/home/jupyter/ai_font/data/processed/seen"
        self.save_path = "experiments1"
        self.bucket_name = 'leo_font'
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_weight_decay = 1e-2
        self.adam_epsilon = 1e-08
        self.max_grad_norm = 1.0
        
        self.seed = 123
        self.experiment_name = "vanilla_fontdiffuser_training_phase_1"
        self.resolution=96
        self.style_image_size=96
        self.content_image_size=96 
        self.content_encoder_downsample_size=3 
        self.channel_attn=True 
        self.content_start_channel=64 
        self.style_start_channel=64 
        self.train_batch_size=8
        self.perceptual_coefficient=0.01 
        self.offset_coefficient=0.5 
        self.max_train_steps=440000 
        self.ckpt_interval=40000 
        self.gradient_accumulation_steps=1 
        self.log_interval=50 
        self.learning_rate=1e-4 
        self.lr_scheduler="linear" 
        self.lr_warmup_steps=10000 
        self.drop_prob=0.1 
        self.mixed_precision="no"
