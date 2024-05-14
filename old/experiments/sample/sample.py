import os
import cv2
import time
import random
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms

import sys
import importlib

def call_module(nm, path):
    spec = importlib.util.spec_from_file_location(nm, path)
    foo = importlib.util.module_from_spec(spec)
    sys.modules[nm] = foo
    spec.loader.exec_module(foo)
    return foo


def image_process(args, content_image, style_image):
    ## Dataset transform
    content_inference_transforms = transforms.Compose(
        [transforms.Resize(args.content_image_size, \
                            interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])
    style_inference_transforms = transforms.Compose(
        [transforms.Resize(args.style_image_size, \
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    content_image = content_inference_transforms(content_image)[None, :]
    style_image = style_inference_transforms(style_image)[None, :]

    return content_image, style_image

def load_fontdiffuser_pipeline(args, module_fd, modelpaths):
    
    model = call_module('dataset', f"{module_fd}/model.py")
    FontDiffuserDPMPipeline = model.FontDiffuserDPMPipeline
    FontDiffuserModelDPM = model.FontDiffuserModelDPM

    build = call_module('build', f"{module_fd}/build.py")
    build_ddpm_scheduler = build.build_ddpm_scheduler
    build_unet = build.build_unet
    build_content_encoder = build.build_content_encoder
    build_style_encoder = build.build_style_encoder
    
    # Load the model state_dict
    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(modelpaths['unet']))
    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(modelpaths['style_encoder']))
    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(modelpaths['content_encoder']))
    model = FontDiffuserModelDPM(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder)
    model.to(args.device)
    print("Loaded the model state_dict successfully!")

    # Load the training ddpm_scheduler.
    train_scheduler = build_ddpm_scheduler(args=args)
    print("Loaded training DDPM scheduler sucessfully!")

    # Load the DPM_Solver to generate the sample.
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )
    print("Loaded dpm_solver pipeline sucessfully!")

    return pipe


def sampling(args, pipe, content_image, style_image, verbose=True):
    
    content_image, style_image = image_process(args=args, 
                                                content_image=content_image, 
                                                style_image=style_image)
    with torch.no_grad():
        content_image = content_image.to(args.device)
        style_image = style_image.to(args.device)
        if verbose:
            print(f"Sampling by DPM-Solver++ ......")
        start = time.time()
        images = pipe.generate(
            content_images=content_image,
            style_images=style_image,
            batch_size=1,
            order=args.order,
            num_inference_step=args.num_inference_steps,
            content_encoder_downsample_size=args.content_encoder_downsample_size,
            t_start=args.t_start,
            t_end=args.t_end,
            dm_size=args.content_image_size,
            algorithm_type=args.algorithm_type,
            skip_type=args.skip_type,
            method=args.method,
            correcting_x0_fn=args.correcting_x0_fn)
        end = time.time()
        return images[0]

