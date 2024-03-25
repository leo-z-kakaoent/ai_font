import os
import pandas as pd
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FontDataset(Dataset):

    def __init__(self, path, content_name="gulim"):
        super().__init__()
        self.path = path
        self.resolution = 128
        self.content_name = content_name
        self.all_files = [path + f for f in os.listdir(path) if ".png" in f]
        self.all_korean_letters = pd.read_parquet(path + "all_korean.parquet")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        target_img_path = self.all_files[index]
        target_img_name = target_img_path.replace(".png", "").split('__')
        style = target_img_name[0]
        content = target_img_name[1]

        content_img_paths = self.path + self.content_name + "__" + content + ".png"
        style_img_paths = [f for f in self.all_files if (style in f) & (target_img_path != f)]
        style_img_path = random.choice(style_img_paths)

        target_img = self.transform(Image.open(target_img_path).convert("RGB"))
        content_img = self.transform(Image.open(content_img_paths).convert("RGB"))
        style_img = self.transform(Image.open(style_img_path).convert("RGB"))

        return {
            "target": target_img,
            "content": content_img,
            "style": style_img,
            "content_name": content,
            "style_name": style,
        }

class SCRDataset(FontDataset):
    def __init__(self, path, content_name="gulim", scr_fd=None, n_neg=4):
        super().__init__(path, content_name)
        self.scr_fd = scr_fd
        self.n_neg = n_neg
        self.scr_files = [path + f for f in os.listdir(self.scr_fd) if ".png" in f]

    def __getitem__(self, index):
        result_dict = super().__getitem__(index)
        positive_img_path = [f for f in self.scr_files if (result_dict["style_name"] not in f) & (result_dict["content_name"] in f)]
        negative_img_path = [f for f in self.scr_files if (result_dict["style_name"] in f) & (result_dict["content_name"] not in f)]
        positive_img = self.transform(Image.open(random.choice(positive_img_path)).convert("RGB"))
        for i in range(self.n_neg):
            neg_image = self.transform(Image.open(random.choice(negative_img_path)).convert("RGB"))
            if i == 0:
                neg_images = neg_image[None, :, :, :]
            else:
                neg_images = torch.cat([neg_images, neg_image[None, :, :, :]], dim=0)
        result_dict["positive"] = positive_img
        result_dict["negative"] = neg_images