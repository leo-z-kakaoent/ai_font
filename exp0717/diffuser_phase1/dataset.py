import os
import random
import copy
from PIL import Image

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class CollateFN(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        batched_data = {}

        for k in batch[0].keys():
            batch_key_data = [ele[k] for ele in batch]
            if isinstance(batch_key_data[0], torch.Tensor):
                batch_key_data = torch.stack(batch_key_data)
            batched_data[k] = batch_key_data
        
        return batched_data


def get_normal_transform(resolution):
    normal_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    return normal_transform

def get_nonorm_transform(resolution):
    nonorm_transform =  transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])
    return nonorm_transform


class FontDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.scr = args.scr
        self.resolution = args.resolution # default
        self.num_neg = args.num_neg
        self.fonts = sorted([f for f in os.listdir(self.args.target_path) if ".ipy" not in f])
        self.fontdict = {}
        for font in self.fonts:
            self.fontdict[font] = sorted([f.replace(".png","").split("__")[-1] for f in os.listdir(f"{self.args.target_path}/{font}/") if f.endswith(".png")])
        
        self.transforms = get_normal_transform(self.resolution)
        self.nonorm_transforms = get_nonorm_transform(self.resolution)

    def __getitem__(self, index):

        font = self.fonts[index]
        content = random.choice(self.fontdict[font])

        content_img_path = f"{self.args.content_path}/{self.args.content_font}/{self.args.content_font}__{content}.png"
        style_img_path = f"{self.args.style_path}/{font}/{font}__{content}.png"
        target_img_path = f"{self.args.target_path}/{font}/{font}__{content}.png"
        
        content_image = self.transforms(Image.open(content_img_path).convert('RGB'))
        style_image = self.transforms(Image.open(style_img_path).convert('RGB'))
        target_image = Image.open(target_img_path).convert('RGB')
        nonorm_target_image = self.nonorm_transforms(target_image)
        target_image = self.transforms(target_image)

        sample = {
            "content_image": content_image,
            "style_image": style_image,
            "target_image": target_image,
            "target_image_path": target_img_path,
            "nonorm_target_image": nonorm_target_image}
        
        if self.scr:
            # Get neg image from the different style of the same content
            neg_names = [f for f in self.target_images if (style not in f)&("__"+content in f)]
            choose_neg_names = []
            for i in range(self.num_neg):
                choose_neg_names.append(random.choice(neg_names))

            # Load neg_images
            for i, neg_name in enumerate(choose_neg_names):
                neg_image = Image.open(neg_name).convert("RGB")
                neg_image = self.transforms(neg_image)
                if i == 0:
                    neg_images = neg_image[None, :, :, :]
                else:
                    neg_images = torch.cat([neg_images, neg_image[None, :, :, :]], dim=0)
            sample["neg_images"] = neg_images

        return sample

    def __len__(self):
        return len(self.fonts)
