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
from fixed_fd.dataset import FontDataset, CollateFN

from fixed_fd.model import FontDiffuserModel
from fixed_fd.criterion import ContentPerceptualLoss
from fixed_fd.build import build_unet, build_style_encoder, build_content_encoder, build_ddpm_scheduler
from fixed_fd.args import TrainPhase1Args
from fixed_fd.utils import save_args_to_yaml, x0_from_epsilon, reNormalize_img, normalize_mean_std

args = TrainPhase1Args()
set_seed(args.seed)
accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision)

unet = build_unet(args=args)
style_encoder = build_style_encoder(args=args)
content_encoder = build_content_encoder(args=args)
noise_scheduler = build_ddpm_scheduler(args)

# unet.load_state_dict(torch.load("data/m40216/unet.pth"))
# style_encoder.load_state_dict(torch.load("data/m40216/style_encoder.pth"))
# content_encoder.load_state_dict(torch.load("data/m40216/content_encoder.pth"))

model = FontDiffuserModel(
    unet=unet,
    style_encoder=style_encoder,
    content_encoder=content_encoder)

perceptual_loss = ContentPerceptualLoss()

train_font_dataset = FontDataset(path=args.path)
train_dataloader = torch.utils.data.DataLoader(
    train_font_dataset, 
    shuffle=True, 
    batch_size=args.train_batch_size, 
    collate_fn=CollateFN())

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon)

lr_scheduler = get_scheduler(
    args.lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
    num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,)

model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler)

if accelerator.is_main_process:
    accelerator.init_trackers(args.experience_name)
progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
progress_bar.set_description("Steps")
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
global_step = 0
for epoch in range(num_train_epochs):
    train_loss = 0.0
    for step, samples in enumerate(train_dataloader):
        model.train()
        content_images = samples["content_image"]
        style_images = samples["style_image"]
        target_images = samples["target_image"]
        nonorm_target_images = samples["nonorm_target_image"]

        with accelerator.accumulate(model):
            # Sample noise that we'll add to the samples
            noise = torch.randn_like(target_images)
            bsz = target_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=target_images.device)
            timesteps = timesteps.long()

            # Add noise to the target_images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_target_images = noise_scheduler.add_noise(target_images, noise, timesteps)

            # Classifier-free training strategy
            context_mask = torch.bernoulli(torch.zeros(bsz) + args.drop_prob)
            for i, mask_value in enumerate(context_mask):
                if mask_value==1:
                    content_images[i, :, :, :] = 1
                    style_images[i, :, :, :] = 1

            # Predict the noise residual and compute loss
            noise_pred, offset_out_sum = model(
                x_t=noisy_target_images, 
                timesteps=timesteps, 
                style_images=style_images,
                content_images=content_images,
                content_encoder_downsample_size=args.content_encoder_downsample_size)
            diff_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            offset_loss = offset_out_sum / 2

            # output processing for content perceptual loss
            pred_original_sample_norm = x0_from_epsilon(
                scheduler=noise_scheduler,
                noise_pred=noise_pred,
                x_t=noisy_target_images,
                timesteps=timesteps)
            pred_original_sample = reNormalize_img(pred_original_sample_norm)
            norm_pred_ori = normalize_mean_std(pred_original_sample)
            norm_target_ori = normalize_mean_std(nonorm_target_images)
            percep_loss = perceptual_loss.calculate_loss(
                generated_images=norm_pred_ori,
                target_images=norm_target_ori,
                device=target_images.device)

            loss = diff_loss + \
                    args.perceptual_coefficient * percep_loss + \
                        args.offset_coefficient * offset_loss

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps

            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if global_step % 10000 == 0:
            torch.save(unet.state_dict(), "data/model/unet_%s.pth"%str(global_step))
            torch.save(style_encoder.state_dict(), "data/model/style_encoder_%s.pth"%str(global_step))
            torch.save(content_encoder.state_dict(), "data/model/content_encoder_%s.pth"%str(global_step))
            
        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            accelerator.log({"train_loss": train_loss}, step=global_step)
            train_loss = 0.0
            
        # Quit
        if global_step >= args.max_train_steps:
            break

accelerator.end_training()
