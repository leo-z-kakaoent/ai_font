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
        self.path = args.datapath
        self.scr = args.scr
        self.resolution = args.resolution # default
        self.num_neg = args.num_neg
        self.ces = args.content_encoding_size
        self.letter_mapper_a = pd.read_pickle(f"{self.path}/pickle/letter_mapper_a.pickle")
        self.letter_mapper_b = pd.read_pickle(f"{self.path}/pickle/letter_mapper_b.pickle")
        self.font_mapper = pd.read_pickle(f"{self.path}/pickle/font_mapper.pickle")
        self.letter_mapper_ab = self.letter_mapper_a.similar + self.letter_mapper_b.similar
        self.fonts = self.font_mapper.index
        
        self.ak = self.get_all_korean() 
        
        self.transforms = get_normal_transform(self.resolution)
        self.nonorm_transforms = get_nonorm_transform(self.resolution)

    def __getitem__(self, index):
        font = self.fonts[index]
        contents = copy.deepcopy(self.font_mapper.loc[font])
        
        target_content = contents.pop(random.randint(0, len(contents)-1))
        content_img_path = f"{self.path}/train/pngs/{self.args.content_font}__{target_content}.png"
        
        style_content = contents.pop(random.randint(0, len(contents)-1))
        style_img_path = f"{self.path}/train/pngs/{font}__{style_content}.png"

        target_img_path = f"{self.path}/train/pngs/{font}__{target_content}.png"
        
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
    
    def get_all_korean(self):

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
