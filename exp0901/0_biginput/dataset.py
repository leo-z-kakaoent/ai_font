import os
import random
import copy
from PIL import Image

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
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


def get_all_korean():

    def nextKorLetterFrom(letter):
        lastLetterInt = 15572643
        if not letter:
            return '가'
        a = letter
        b = a.encode('utf8')
        c = int(b.hex(), 16)

        if c == lastLetterInt:
            return False

        d = hex(c + 1)
        e = bytearray.fromhex(d[2:])

        flag = True
        while flag:
            try:
                r = e.decode('utf-8')
                flag = False
            except UnicodeDecodeError:
                c = c+1
                d = hex(c)
                e = bytearray.fromhex(d[2:])
        return e.decode()

    returns = []
    flag = True
    k = ''
    while flag:
        k = nextKorLetterFrom(k)
        if k is False:
            flag = False
        else:
            returns.append(k)
    return returns

def get_font_mapper(path, tag):
    ak = get_all_korean()
    fonts = os.listdir(f"{path}/train_whole")
    font_mapper = defaultdict(list)
    pbar = tqdm(fonts)
    for font in pbar:
        for letter in ak:
            target_exists = os.path.exists(f"{path}/train_whole/{font}/{font}__{tag}__{letter}.png")
            style_exists = os.path.exists(f"{path}/train_assembled/{font}/{font}__{tag}__{letter}.png")
            if target_exists & style_exists:
                font_mapper[font].append(letter)
        pbar.set_postfix(font=font)
    return font_mapper


class FontDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.path = args.datapath
        self.scr = args.scr
        self.resolution = args.resolution # default
        self.num_neg = args.num_neg
        self.tags = ['closing','dilate','erode']
        self.font_mapper = get_font_mapper(self.path, 'closing')
        self.fonts = [k for k in self.font_mapper.keys() if k != "플레이브밤비"]
        self.transforms = get_normal_transform(self.resolution)
        self.nonorm_transforms = get_nonorm_transform(self.resolution)

    def __getitem__(self, index):

        font = self.fonts[index]
        tag = random.choice(self.tags)
        
        content = random.choice(self.font_mapper[font])
        
        style_img_path = f"{self.path}/train_assembled/{font}/{font}__{tag}__{content}.png"
        content_img_path = f"{self.path}/train_whole/{self.args.content_font}/{self.args.content_font}__closing__{content}.png"
        target_img_path = f"{self.path}/train_whole/{font}/{font}__{tag}__{content}.png"
        
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
            i = 0
            while i < self.num_neg:
                f = random.choice(self.fonts)
                t = self.tag
                neg_path = f"{self.path}/train_whole/{f}/{f}__{t}__{content}.png"
                if (f != font) & os.path.exists(neg_path):
                    neg_image = Image.open(neg_path).convert("RGB")
                    neg_image = self.transforms(neg_image)
                    if i == 0:
                        neg_images = neg_image[None, :, :, :]
                    else:
                        neg_images = torch.cat([neg_images, neg_image[None, :, :, :]], dim=0)
                    i += 1
            sample["neg_images"] = neg_images

        return sample
    def __len__(self):
        return len(self.fonts)
