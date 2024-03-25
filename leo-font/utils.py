import torch
from datetime import datetime
from PIL import Image

def save_package(img_dict, model_dict, path, prefix):
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    for k,v in img_dict.items():
        p = f"{path}/{prefix}__{k}_{current_time}.png"
        Image.fromarray(v).save(p)
        print(f"Saved: {p}")
    for k,v in model_dict.items():
        p = f"{path}/{prefix}__{k}_{current_time}.pth"
        torch.save(v, p)
        print(f"Saved: {p}")