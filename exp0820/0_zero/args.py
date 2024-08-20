
class SampleArgs:
    def __init__(self):
        
        self.bucket_name = "leo_font"
        self.content_encoder_path=f"exp0604/phase2/phase_2__1000000__content_encoder_30000.pth"
        self.style_encoder_path=f"exp0604/phase2/phase_2__1000000__style_encoder_30000.pth"
        self.unet_path=f"exp0604/phase2/phase_2__1000000__unet_30000.pth"
        self.datapath = "/home/jupyter/ai_font/data"
        self.batchsize = 32
        self.savefd = f"/home/jupyter/ai_font/data/report/zero"
        self.tag = 'zero'
        
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