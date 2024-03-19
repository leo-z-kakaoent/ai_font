import os
import random
from PIL import Image

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from fixed_fd.scr import SCR
from diffusers.optimization import get_scheduler
from accelerate import Accelerator


# Sample, Positive, Negative. By Style
class SCRDataset(Dataset):

    def __init__(self, path, num_neg=8):
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
    
scr_ds = SCRDataset(path="data/raw/")
scr_dl = DataLoader(scr_ds, shuffle=True, batch_size=32+16, num_workers=8)
scr_model = SCR()
optimizer = torch.optim.AdamW(scr_model.parameters(), lr=1e-5)
epoch = 1000
save_fd = "data/model/"

accelerator = Accelerator()
lr_scheduler = get_scheduler(name="linear", optimizer=optimizer,num_warmup_steps=10, num_training_steps=1000)
scr_model, optimizer, scr_dl, lr_scheduler = accelerator.prepare(scr_model, optimizer, scr_dl, lr_scheduler)


max_train_steps = epoch * len(scr_dl)
progress_bar = tqdm(range(max_train_steps))
progress_bar.set_description("Steps")
lossdicts = []
for epoch_i in range(epoch):
    losses = []
    for step, x in enumerate(scr_dl):
        with accelerator.accumulate(scr_model):
            sample_img, pos_img, neg_imgs = x
            sample_emb, pos_emb, neg_emb = scr_model(sample_img, pos_img, neg_imgs)
            loss = scr_model.calculate_nce_loss(sample_emb, pos_emb, neg_emb)

            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            losses.append(loss.detach().cpu().numpy())
            progress_bar.update(1)
            progress_bar.set_postfix(loss=np.mean(losses))
    lossdicts.append({"loss":np.mean(losses)})
    pd.DataFrame().from_dict(lossdicts).to_csv(save_fd+"loss.csv")
    torch.save(scr_model.state_dict(), save_fd+"m_%s.pth"%str(epoch_i))
accelerator.end_training()