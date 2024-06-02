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

    def __init__(self, args, sampling_font_index):
        super().__init__()
        self.args = args
        self.path = args.datapath
        self.resolution = args.resolution # default
        self.testmap = pd.read_pickle("/home/jupyter/ai_font/data/test/testmapdf.pickle")
        self.fonts = np.unique(self.testmap['font'])
        self.testmap = self.testmap.loc[np.isin(self.testmap['font'].values,self.fonts[sampling_font_index])]
        self.content_font = '시스템 굴림'
        self.transforms = get_normal_transform(self.resolution)
        
        
    def __len__(self):
        return len(self.testmap)
    
    def __getitem__(self, index):
        
        arow = self.testmap.iloc[index]
        # B 가 content, A가 style, writerId는 A 거를 가져옴
        style_font = arow.font
        style_content = arow.double[0] if len(arow.double)>0 else arow.single[0]
        content_font = self.content_font
        content_content = arow.letter
        
        content_path = f"{self.path}/train/pngs/{content_font}__{content_content}.png"
        style_path = f"{self.path}/test/pngs/{style_font}__{style_content}.png"
        
        sample = {
            "content_img": self.transforms(Image.open(content_path).convert('RGB')),
            "style_img": self.transforms(Image.open(style_path).convert('RGB')),
            "font": style_font,
            "content": content_content,
        }
        return sample

def get_normal_transform(resolution):
    normal_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    return normal_transform

