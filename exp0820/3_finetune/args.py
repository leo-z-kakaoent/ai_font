

class TrainPhase2Args:
    def __init__(self):
        
        self.seed=123
        self.bucket_name = "leo_font"
        self.experiment_name="finetune"
        self.savepath = "exp0820/finetune"
        self.datapath = "/home/jupyter/ai_font/data"
        self.phase_2 = True
        self.scr = True
        self.scr_path="exp0514/scr/scr__440000.pth"
        self.content_encoder_path="exp0604/phase1/phase1__content_encoder_1000000.pth"
        self.style_encoder_path="exp0604/phase1/phase1__style_encoder_1000000.pth"
        self.unet_path="exp0604/phase1/phase1__unet_1000000.pth"
        self.content_font = '시스템굴림'
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
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_weight_decay = 1e-2
        self.adam_epsilon = 1e-08
        self.max_grad_norm = 1.0

