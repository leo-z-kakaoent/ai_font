import numpy as np
import pandas as pd


class SampleArgs:
    def __init__(self):
        
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
        
        self.easy = "가너디로마버소우즈치카토퍼후"
        self.mid = "갹넌됻래몌벼슈양쟈챼켴텉픞핳"
        self.hard = "겱냙뎳랛몊볍숎융쟧츣캷툛펆햙"
        
        self.seens = [l for i,l in enumerate(self.easy) if i % 2 == 1]
        self.seens += [l for i,l in enumerate(self.mid) if i % 2 == 0]
        self.seens += [l for i,l in enumerate(self.hard) if i % 2 == 1]

        self.unseens = [l for i,l in enumerate(self.easy) if i % 2 == 0]
        self.unseens += [l for i,l in enumerate(self.mid) if i % 2 == 1]
        self.unseens += [l for i,l in enumerate(self.hard) if i % 2 == 0]

        self.test_fonts = ['twice dahyun_4','twice nayeon_6','UhBee Sunhong','UhBee Howl','SeoulHangang Jang B']
        
        self.data_fd = "/home/jupyter/ai_font/data/zipfiles/raw/size96"
        self.model_fd = "/home/jupyter/ai_font/data/model"
        self.allkorean = pd.read_parquet(f"{self.data_fd}/seen/all_korean.parquet")
        
        self.seens_ids = [np.where(self.allkorean.values==l)[0].item() for l in self.seens]
        self.unseens_ids = [np.where(self.allkorean.values==l)[0].item() for l in self.unseens]
        