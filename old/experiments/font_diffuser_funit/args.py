
class SampleArgs:
    def __init__(self):
        self.ckpt_dir="data/m40216"
        self.demo=True
        self.controlnet=False
        self.character_input=False
        self.content_character=None
        self.content_image_path="data/r40202/pngs/gulim__1.png"
        self.style_image_path='data/r40202/pngs/twice momo_6__507.png'
        self.save_image="store_true"
        self.save_image_dir="data/f40219/"
        self.device="cuda:0"
        self.ttf_path="data/r40202/ttfs/gulim.ttf"
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

class TrainPhase1Args:
    def __init__(self):
        
        self.unet_channels=(64, 128, 256, 512,)
        self.beta_scheduler="scaled_linear"
        self.path = "/home/jupyter/ai_font/data/zipfiles/raw/size96/seen/"
        self.save_path = "experiments"
        self.bucket_name = 'leo_font'
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_weight_decay = 1e-2
        self.adam_epsilon = 1e-08
        self.max_grad_norm = 1.0
        
        self.seed = 123
        self.experiment_name = "funit_fontdiffuser_training_phase_1"
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
        
        
class TrainPhase2Args:
    def __init__(self):
        
        self.seed=123
        self.experience_name="FontDiffuser_training_phase_2"
        self.phase_2 = True
        self.scr_path="data/m40216/scr_99.pth"
        self.content_encoder_path="data/m40216/content_encoder_430000.pth"
        self.style_encoder_path="data/m40216/style_encoder_430000.pth"
        self.unet_path="data/m40216/unet_430000.pth"
        
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
        self.max_train_steps=30000
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
        self.path = "data/r40202/"
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_weight_decay = 1e-2
        self.adam_epsilon = 1e-08
        self.max_grad_norm = 1.0
        