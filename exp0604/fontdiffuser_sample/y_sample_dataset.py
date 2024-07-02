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
        self.path = args.datapath
        self.resolution = args.resolution # default
        self.ak = get_all_korean()
        self.content_font = '시스템굴림'
        self.target_font = target_font
        self.transforms = get_normal_transform(self.resolution)
        self.target_letters = [k for k in self.ak if os.path.exists(f"{self.path}/test_assembled/{self.target_font}/{self.target_font}__closing__{k}.png") & os.path.exists(f"{self.path}/test/processed/{self.target_font}/{self.target_font}__closing__{k}.png")]
        
    def __len__(self):
        return len(self.target_letters)
    
    def __getitem__(self, index):
        
        content = self.target_letters[index]
        content_path = f"{self.path}/train/{self.content_font}/{self.content_font}__closing__{content}.png"
        style_path = f"{self.path}/test_assembled/{self.target_font}/{self.target_font}__closing__{content}.png"
        
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
