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
        self.all_files = [path + "pngs/" + f for f in os.listdir(path + "pngs/") if ".png" in f]
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

        content_img_paths = self.path + "pngs/" + style + "__" + content + ".png"
        style_img_paths = [f for f in self.all_files if (style in f) & (target_img_path != f)]
        style_img_path = random.choice(style_img_paths)

        target_img = self.transform(Image.open(target_img_path).convert("RGB"))
        content_img = self.transform(Image.open(content_img_paths).convert("RGB"))
        style_img = self.transform(Image.open(style_img_path).convert("RGB"))

        return {
            "target": target_img,
            "content": content_img,
            "style": style_img,
        }
