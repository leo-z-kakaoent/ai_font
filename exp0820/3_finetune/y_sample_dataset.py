import os
import random
import copy
from PIL import Image

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class SamplingDataset(Dataset):

    def __init__(self, args, target_font):
        super().__init__()
        self.args = args
        self.resolution = args.resolution # default
        self.transforms = get_normal_transform(self.resolution)
        self.target_font = target_font
        self.target_letters = [k[-5] for k in os.listdir(f"{self.args.stylefd}/{self.target_font}") if k.endswith(".png")]
        
    def __len__(self):
        return len(self.target_letters)
    
    def __getitem__(self, index):
        
        content = self.target_letters[index]
        content_path = f"{self.args.contentfd}/{self.args.content_font}/{self.args.content_font}__closing__{content}.png"
        style_path = f"{self.args.stylefd}/{self.target_font}/{self.target_font}__closing__{content}.png"
        
        sample = {
            "content_img": self.transforms(Image.open(content_path).convert('RGB')),
            "style_img": self.transforms(Image.open(style_path).convert('RGB')),
            "content": content,
        }
        return sample

def get_normal_transform(resolution):
    normal_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    return normal_transform
