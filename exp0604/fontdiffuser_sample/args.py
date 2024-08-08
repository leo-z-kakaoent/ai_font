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
        self.savepath = "exp0604/phase2"
        self.datapath = "/home/jupyter/ai_font/data"
        self.scr = True
        self.num_neg = 4
        self.experiment_name = "phase2"
        self.resolution=96
        self.content_font = '시스템 굴림'
        self.content_encoding_size = 12
        
        self.seed=123
        self.phase_2 = True
        self.scr_path="exp0514/scr/scr__440000.pth"
        self.content_encoder_path="exp0514/phase1/phase1__content_encoder_1000000.pth"
        self.style_encoder_path="exp0514/phase1/phase1__style_encoder_1000000.pth"
        self.unet_path="exp0514/phase1/phase1__unet_1000000.pth"
        
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
    def __init__(self, itern):
        
        self.bucket_name = "leo_font"
        self.content_encoder_path=f"exp0604/phase2/phase_2__1000000__content_encoder_{itern}.pth"
        self.style_encoder_path=f"exp0604/phase2/phase_2__1000000__style_encoder_{itern}.pth"
        self.unet_path=f"exp0604/phase2/phase_2__1000000__unet_{itern}.pth"
        self.datapath = "/home/jupyter/ai_font/data"
        self.batchsize = 32
        self.savefd = f"/home/jupyter/ai_font/data/reports/fontdiffuser/phase2/i{itern}"
        self.tag = 'fontdiffuser_phase2'
        
        # leo_font/exp0604/phase2/phase_2__1000000__content_encoder_0.pth
        
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