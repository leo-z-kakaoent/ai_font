from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import sys
import importlib

def call_module(nm, path):
    spec = importlib.util.spec_from_file_location(nm, path)
    foo = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = foo
    spec.loader.exec_module(foo)
    return foo

fd = "/home/jupyter/ai_font/experiments/font_diffuser_funit"
content_encoder = call_module('content_encoder', f"{fd}/content_encoder.py")
ContentEncoder = content_encoder.ContentEncoder

style_encoder = call_module('style_encoder', f"{fd}/style_encoder.py")
StyleEncoder = style_encoder.StyleEncoder

unet = call_module('unet', f"{fd}/unet.py")
UNet = unet.UNet

scr = call_module('scr', f"{fd}/scr.py")
SCR = scr.SCR


def build_unet(args):
    unet = UNet(
        sample_size=args.resolution,
        in_channels=3,
        out_channels=3,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=('DownBlock2D', 
                          'MCADownBlock2D',
                          'MCADownBlock2D', 
                          'DownBlock2D'),
        up_block_types=('UpBlock2D', 
                        'StyleRSIUpBlock2D',
                        'StyleRSIUpBlock2D', 
                        'UpBlock2D'),
        block_out_channels=args.unet_channels, 
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        act_fn='silu',
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=args.style_start_channel * 16,
        attention_head_dim=1,
        channel_attn=args.channel_attn,
        content_encoder_downsample_size=args.content_encoder_downsample_size,
        content_start_channel=args.content_start_channel,
        reduction=32)
    
    return unet


def build_style_encoder(args):
    style_image_encoder = StyleEncoder(
        G_ch=args.style_start_channel,
        resolution=args.style_image_size)
    print("Get CG-GAN Style Encoder!")
    return style_image_encoder


def build_content_encoder(args):
    content_image_encoder = ContentEncoder(
        G_ch=args.content_start_channel,
        resolution=args.content_image_size)
    print("Get CG-GAN Content Encoder!")
    return content_image_encoder


def build_scr(args):
    scr = SCR(
        temperature=args.temperature,
        mode=args.mode,
        image_size=args.scr_image_size)
    print("Loaded SCR module for supervision successfully!")
    return scr


def build_ddpm_scheduler(args):
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule=args.beta_scheduler,
        trained_betas=None,
        variance_type="fixed_small",
        clip_sample=True)
    return ddpm_scheduler