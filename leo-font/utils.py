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

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad