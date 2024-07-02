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

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.path = args.datapath
        self.resolution = args.resolution # default
        self.testmap = pd.read_pickle("/home/jupyter/ai_font/data/pickle/testmap_플레이브밤비.pickle")
        self.content_font = "나눔손글씨가람연꽃"
        self.transforms = get_normal_transform(self.resolution)
        
        
    def __len__(self):
        return len(self.testmap)
    
    def __getitem__(self, index):
        
        arow = self.testmap.iloc[index]
        # B 가 content, A가 style, writerId는 A 거를 가져옴
        A_font = arow.font
        A_content = arow.double[0] if len(arow.double)>0 else arow.single[0]
        B_font = self.content_font
        B_content = arow.letter
        
        A_path = f"{self.path}/test/pngs/{A_content}.png"
        B_path = f"{self.path}/train/{B_font}/{B_font}__{B_content}.png"
        
        A_label = A_content
        B_label = B_content
        
        sample = {
            "A": self.transforms(Image.open(A_path).convert('RGB')),
            "B": self.transforms(Image.open(B_path).convert('RGB')),
            "A_paths": A_path,
            "writerID": A_font,
            "A_label": A_label,
            "B_label": B_label,
            "val": True,
        }
        return sample

def get_normal_transform(resolution):
    normal_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    return normal_transform

