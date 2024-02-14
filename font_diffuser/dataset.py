import os
import random
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

    
# Sample, Positive, Negative. By Style
class FontDataset(Dataset):

    def __init__(self, path, num_neg=4):
        super().__init__()
        self.path = path
        self.resolution = 96 # default
        self.num_neg = num_neg
        self.all_files = [path+"pngs/"+f for f in os.listdir(path+"pngs/") if ".png" in f]
        self.all_korean_letters = pd.read_parquet(path+"all_korean.parquet")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
    def __len__(self):
        return len(self.all_files)
        
    def __getitem__(self, index):
        sample_img_path = self.all_files[index]
        sample_img_name = sample_img_path.replace(".png","").split('__')
        style = sample_img_name[0]
        
        pos_img_paths = [f for f in self.all_files if (style in f) & (sample_img_path != f)]
        pos_img_path = random.choice(pos_img_paths)
        
        sample_img = self.transform(Image.open(sample_img_path).convert("RGB"))
        pos_img = self.transform(Image.open(pos_img_path).convert("RGB"))
        
        neg_imgs = []
        neg_img_paths = [f for f in self.all_files if (style not in f) & ("__%s"%sample_img_name[1] in f)]
        for _ in range(self.num_neg):
            neg_img_path = random.choice(neg_img_paths)
            neg_imgs.append(self.transform(Image.open(neg_img_path).convert("RGB")))
        
        # sample_img = Image.open(sample_img_path).convert("RGB")
        # pos_img = Image.open(pos_img_path).convert("RGB")
        # neg_img = Image.open(neg_img_path).convert("RGB")
        
        return sample_img, pos_img, torch.stack(neg_imgs)
    
    
import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def get_normal_transform():
    normal_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    return normal_transform

def get_nonorm_transform():
    nonorm_transform =  transforms.Compose([transforms.ToTensor()])
    return nonorm_transform


class FontDataset(Dataset):
    def __init__(self, path, phase, num_neg=4, scr=False):
        super().__init__()
        self.path = path
        self.num_neg = num_neg
        self.all_files = [path+"pngs/"+f for f in os.listdir(path+"pngs/") if ".png" in f]
        self.all_korean_letters = pd.read_parquet(path+"all_korean.parquet")
        self.phase = phase
        self.scr = scr
        if self.scr:
            self.num_neg = nug_neg
        
        self.transforms = get_normal_transform()
        self.nonorm_transforms = get_nonorm_transform()

    def __getitem__(self, index):
        target_image_path = self.target_images[index]
        target_image_name = target_image_path.split('/')[-1]
        style, content = target_image_name.split('.')[0].split('+')
        
        # Read content image
        content_image_path = f"{self.root}/{self.phase}/ContentImage/{content}.jpg"
        content_image = Image.open(content_image_path).convert('RGB')

        # Random sample used for style image
        images_related_style = self.style_to_images[style].copy()
        images_related_style.remove(target_image_path)
        style_image_path = random.choice(images_related_style)
        style_image = Image.open(style_image_path).convert("RGB")
        
        # Read target image
        target_image = Image.open(target_image_path).convert("RGB")
        nonorm_target_image = self.nonorm_transforms(target_image)

        if self.transforms is not None:
            content_image = self.transforms[0](content_image)
            style_image = self.transforms[1](style_image)
            target_image = self.transforms[2](target_image)
        
        sample = {
            "content_image": content_image,
            "style_image": style_image,
            "target_image": target_image,
            "target_image_path": target_image_path,
            "nonorm_target_image": nonorm_target_image}
        
        if self.scr:
            # Get neg image from the different style of the same content
            style_list = list(self.style_to_images.keys())
            style_index = style_list.index(style)
            style_list.pop(style_index)
            choose_neg_names = []
            for i in range(self.num_neg):
                choose_style = random.choice(style_list)
                choose_index = style_list.index(choose_style)
                style_list.pop(choose_index)
                choose_neg_name = f"{self.root}/train/TargetImage/{choose_style}/{choose_style}+{content}.jpg"
                choose_neg_names.append(choose_neg_name)

            # Load neg_images
            for i, neg_name in enumerate(choose_neg_names):
                neg_image = Image.open(neg_name).convert("RGB")
                if self.transforms is not None:
                    neg_image = self.transforms[2](neg_image)
                if i == 0:
                    neg_images = neg_image[None, :, :, :]
                else:
                    neg_images = torch.cat([neg_images, neg_image[None, :, :, :]], dim=0)
            sample["neg_images"] = neg_images

        return sample

    def __len__(self):
        return len(self.target_images)