import os
import math
import time
import yaml
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

from accelerate import Accelerator
from accelerate.utils import set_seed

from diffusers.optimization import get_scheduler
from font_diffuser.dataset import FontDataset, CollateFN

from font_diffuser.model import FontDiffuserModel
from font_diffuser.criterion import ContentPerceptualLoss
from font_diffuser.build import build_unet, build_style_encoder, build_content_encoder, build_ddpm_scheduler
from font_diffuser.args import TrainPhase1Args
from font_diffuser.utils import save_args_to_yaml, x0_from_epsilon, reNormalize_img, normalize_mean_std


args = TrainPhase1Args()
set_seed(args.seed)
accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision)

unet = build_unet(args=args)
style_encoder = build_style_encoder(args=args)
content_encoder = build_content_encoder(args=args)
noise_scheduler = build_ddpm_scheduler(args)

model = FontDiffuserModel(
    unet=unet,
    style_encoder=style_encoder,
    content_encoder=content_encoder)

perceptual_loss = ContentPerceptualLoss()

train_font_dataset = FontDataset(path=args.path, phase='train')
train_dataloader = torch.utils.data.DataLoader(
    train_font_dataset, 
    shuffle=True, 
    batch_size=args.train_batch_size, 
    collate_fn=CollateFN())

step, samples = next(enumerate(train_dataloader))

model.train()
content_images = samples["content_image"].cuda()
style_images = samples["style_image"].cuda()
target_images = samples["target_image"].cuda()
nonorm_target_images = samples["nonorm_target_image"].cuda()

a = style_images[[1]]
try:
    s = style_encoder.cuda()
    s(a)
except Error as error:
    print(error)
