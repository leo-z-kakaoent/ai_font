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



class CGGANDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.path = args.datapath
        self.resolution = args.resolution # default
        self.letter_mapper_a = pd.read_pickle(f"{self.path}/pickle/letter_mapper_a.pickle")
        self.letter_mapper_b = pd.read_pickle(f"{self.path}/pickle/letter_mapper_b.pickle")
        self.font_mapper = pd.read_pickle(f"{self.path}/pickle/font_mapper.pickle")
        self.letter2font = pd.read_pickle(f"{self.path}/pickle/letter2font.pickle")
        self.letter_mapper_ab = self.letter_mapper_a.similar + self.letter_mapper_b.similar
        self.fonts = self.font_mapper.index
        
        self.ak = self.get_all_korean()
        self.transforms = get_normal_transform(self.resolution)
        self.tags = ["closing","erode","dilate"]

    def __getitem__(self, index):
        # B 가 content, A가 style, writerId는 A 거를 가져옴
        A_font = self.fonts[index]
        A_tag = random.choice(self.tags)
        B_tag = random.choice(self.tags)
        
        contents = self.font_mapper.loc[A_font]
        B_content = random.choice(contents)
        # print(contents)
        # print(self.letter_mapper_ab[B_content])
        A_pool = set(contents).intersection(set(self.letter_mapper_ab[B_content])) 
        A_pool.discard(B_content)
        A_content = random.choice(list(A_pool))
        
        styles = [x for x in self.letter2font.loc[B_content] if x != A_font]
        B_font = random.choice(styles)
        
        A_path = f"{self.path}/train/{A_font}/{A_font}__{A_tag}__{A_content}.png"
        B_path = f"{self.path}/train/{B_font}/{B_font}__{B_tag}__{B_content}.png"
        
        writerID = torch.tensor(index)
        
        A_label = A_content
        B_label = B_content
        
        A_lexicon = torch.tensor(self.get_lexicon(A_content))
        B_lexicon = torch.tensor(self.get_lexicon(B_content))
        
        sample = {
            "A": self.transforms(Image.open(A_path).convert('RGB')),
            "B": self.transforms(Image.open(B_path).convert('RGB')),
            "A_paths": A_path,
            "writerID": writerID,
            "A_label": A_label,
            "B_label": B_label,
            "root" : self.path,
            "A_lexicon": A_lexicon,
            "B_lexicon": B_lexicon,
        }
        return sample

    def __len__(self):
        return len(self.fonts)
    
    def get_lexicon(self, letter):
        ch1 = (ord(letter) - ord('가'))//588
        ch2 = ((ord(letter) - ord('가')) - (588*ch1)) // 28
        ch3 = (ord(letter) - ord('가')) - (588*ch1) - 28*ch2
        # lengths = [19,21,28]
        padded = (0, ch1+2, ch2+19+2, ch3+19+21+2, 1) 
        return padded
    
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
