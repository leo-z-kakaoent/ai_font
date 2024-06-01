class TrainPhase1Args:
    def __init__(self):
        # My Configs
        self.bucket_name = "leo_font"
        self.savepath = "exp0514/phase1"
        self.datapath = "/home/jupyter/ai_font/data"
        self.scr = False
        self.num_neg = None
        self.experiment_name = "van_phase1"
        self.resolution=96
        self.content_font = '시스템 굴림'
        self.content_encoding_size = 12
        
        # Given
        self.unet_channels=(64, 128, 256, 512,)
        self.beta_scheduler="scaled_linear"
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_weight_decay = 1e-2
        self.adam_epsilon = 1e-08
        self.max_grad_norm = 1.0
        self.seed = 123
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


class TrainPhase2Args:
    def __init__(self):
        
        # My Configs
        self.bucket_name = "leo_font"
        self.savepath = "exp0514/phase2"
        self.datapath = "/home/jupyter/ai_font/data"
        self.scr = True
        self.num_neg = 4
        self.experiment_name = "van_phase2"
        self.resolution=96
        self.content_font = '시스템 굴림'
        self.content_encoding_size = 12
        
        self.seed=123
        self.phase_2 = True
        self.scr_path="exp0514/scr/scr__440000.pth"
        self.content_encoder_path="exp0514/phase1/van_phase1__content_encoder_430000.pth"
        self.style_encoder_path="exp0514/phase1/van_phase1__style_encoder_430000.pth"
        self.unet_path="exp0514/phase1/van_phase1__unet_430000.pth"
        
        self.temperature=0.07
        self.mode="refinement"
        self.nce_layers='0,1,2,3'
        
        self.sc_coefficient=0.01
        self.num_neg=4
        self.resolution=96
        self.style_image_size=96
        self.content_image_size=96
        self.scr_image_size=96
        self.content_encoder_downsample_size=3
        self.channel_attn=True
        self.content_start_channel=64
        self.style_start_channel=64
        self.train_batch_size=8
        self.perceptual_coefficient=0.01
        self.offset_coefficient=0.5
        self.max_train_steps=50000
        self.ckpt_interval=5000
        self.gradient_accumulation_steps=1
        self.log_interval=50
        self.learning_rate=1e-5
        self.lr_scheduler="constant"
        self.lr_warmup_steps=1000
        self.drop_prob=0.1
        self.mixed_precision="no"
        
        self.unet_channels=(64, 128, 256, 512,)
        self.beta_scheduler="scaled_linear"
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_weight_decay = 1e-2
        self.adam_epsilon = 1e-08
        self.max_grad_norm = 1.0

class SampleArgs:
    def __init__(self):
        
        self.bucket_name = "leo_font"
        self.content_encoder_path="exp0514/phase2/van_phase2__content_encoder_30000.pth"
        self.style_encoder_path="exp0514/phase2/van_phase2__style_encoder_30000.pth"
        self.unet_path="exp0514/phase2/van_phase2__unet_30000.pth"
        self.datapath = "/home/jupyter/ai_font/data"
        self.batchsize = 96
        self.savefd = "/home/jupyter/ai_font/data/reports/exp0514/van_fontdiffuser"
        self.tag = 'van_fontdif'
        
        
        
        self.device="cuda:0"
        self.style_image_size = 96
        self.content_image_size = 96
        
        self.resolution = 96
        self.unet_channels=(64, 128, 256, 512,)
        self.style_start_channel=64 
        self.channel_attn=True 
        self.content_encoder_downsample_size=3 
        self.content_start_channel=64 
        self.beta_scheduler="scaled_linear"
        self.model_type="noise"
        self.guidance_type="classifier-free"
        self.guidance_scale=7.5
        self.seed = 123
        
        self.algorithm_type="dpmsolver++"
        self.num_inference_steps=20
        self.method="multistep"
        
        self.order=2
        self.t_start=None
        self.t_end=None
        self.skip_type="time_uniform"
        self.correcting_x0_fn=None