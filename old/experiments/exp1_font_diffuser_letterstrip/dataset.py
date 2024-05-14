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
    def __init__(self, path, num_neg=4, scr=False):
        super().__init__()
        self.path = path
        self.num_neg = num_neg
        self.target_images = [path+f for f in os.listdir(path) if ".png" in f]
        self.all_koreans = pd.read_parquet(path+'all_korean.parquet')
        self.onehot = self.all_koreans[0].apply(self.korean2onehot)
        self.scr = scr
        if self.scr:
            self.num_neg = num_neg
        
        self.transforms = get_normal_transform()
        self.nonorm_transforms = get_nonorm_transform()

    def __getitem__(self, index):
        target_image_path = self.target_images[index]
        target_image_name = target_image_path.split('/')[-1]
        style, content = target_image_name.split(".png")[0].split("__")
        content_encoding = torch.zeros([68,16,16])
        content_encoding[np.where(self.onehot.iloc[int(content)])[0],:,:] = 1
        
        # Read content image
        content_image_path = f"{self.path}gulim__{content}.png"
        content_image = Image.open(content_image_path).convert('RGB')

        # Random sample used for style image
        images_related_style = [f for f in self.target_images if (style in f)&("__"+content not in f)]
        style_image_path = random.choice(images_related_style)
        style_image = Image.open(style_image_path).convert("RGB")
        
        # Read target image
        target_image = Image.open(target_image_path).convert("RGB")
        nonorm_target_image = self.nonorm_transforms(target_image)

        content_image = self.transforms(content_image)
        style_image = self.transforms(style_image)
        target_image = self.transforms(target_image)
        
        sample = {
            "content_image": content_image,
            "style_image": style_image,
            "target_image": target_image,
            "target_image_path": target_image_path,
            "nonorm_target_image": nonorm_target_image,
            "content_encoding": content_encoding}
        
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
        return len(self.target_images)
    
    def korean2onehot(self, letter):
        ch1 = (ord(letter) - ord('가'))//588
        ch2 = ((ord(letter) - ord('가')) - (588*ch1)) // 28
        ch3 = (ord(letter) - ord('가')) - (588*ch1) - 28*ch2
        return torch.from_numpy(np.concatenate([
            np.eye(19)[ch1],
            np.eye(21)[ch2],
            np.eye(28)[ch3],
        ]))