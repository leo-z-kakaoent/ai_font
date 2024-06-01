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
        self.resolution = args.resolution
        self.ces = args.content_encoding_size
        self.testmap = pd.read_pickle(self.args.testmap_path)
        self.fonts = np.unique(self.testmap['font'])
        self.testmap = self.testmap.loc[np.isin(self.testmap['font'].values,self.fonts[sampling_font_index])]
        self.content_font = '시스템 굴림'
        self.transforms = get_normal_transform(self.resolution)
        self.ak = self.get_all_korean() 
        self.onehot = {k:self.korean2onehot(k) for k in self.ak}  
        
        
    def __len__(self):
        return len(self.testmap)
    
    def __getitem__(self, index):
        
        arow = self.testmap.iloc[index]
        # B 가 content, A가 style, writerId는 A 거를 가져옴
        style_font = arow.font
        content_font = self.content_font
        content_content = arow.letter
        if len(arow.double) >= 2:
            style_content0 = arow.double[0]
            style_content1 = arow.double[1]
        elif len(arow.double) == 1:
            style_content0 = arow.double[0]
            style_content1 = arow.single[0]
        else:
            style_content0 = arow.single[0]
            style_content1 = arow.single[1]
        
        content_path = f"{self.path}/train/pngs/{content_font}__{content_content}.png"
        style_path0 = f"{self.path}/test/pngs/{style_font}__{style_content0}.png"
        style_path1 = f"{self.path}/test/pngs/{style_font}__{style_content1}.png"
        content_encoding = torch.zeros([68, self.ces, self.ces])
        content_encoding[np.where(self.onehot[content_content])[0],:,:] = 1
        
        sample = {
            "content_img": self.transforms(Image.open(content_path).convert('RGB')),
            "style_img0": self.transforms(Image.open(style_path0).convert('RGB')),
            "style_img1": self.transforms(Image.open(style_path1).convert('RGB')),
            "content_encoding": content_encoding,
            "font": style_font,
            "content": content_content,
        }
        return sample
    
    def korean2onehot(self, letter):
        ch1 = (ord(letter) - ord('가'))//588
        ch2 = ((ord(letter) - ord('가')) - (588*ch1)) // 28
        ch3 = (ord(letter) - ord('가')) - (588*ch1) - 28*ch2
        return torch.from_numpy(np.concatenate([
            np.eye(19)[ch1],
            np.eye(21)[ch2],
            np.eye(28)[ch3],
        ]))
    
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


def get_normal_transform(resolution):
    normal_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    return normal_transform

