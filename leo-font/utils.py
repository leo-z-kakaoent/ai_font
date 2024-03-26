import torch
import numpy as np
from datetime import datetime
from PIL import Image

def tensor2img(img_tensor):
    imgarray = img_tensor.detach().cpu().numpy().transpose((1,2,0))
    imgarray = ((imgarray + 1) / 2 *255).astype(np.uint8)
    return Image.fromarray(imgarray)

def img_concat(img0, img1, r=128):
    plate = Image.new("RGB", (r*2, r))
    plate.paste(img0, (0,0))
    plate.paste(img1, (r,0))
    return plate

def save_package(img_dict, model_dict, path, prefix):
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    for k,v in img_dict.items():
        p = f"{path}/{prefix}__{k}_{current_time}.png"
        img_concat(tensor2img(v[0]),tensor2img(v[1])).save(p)
        # print(f"Saved: {p}")
    for k,v in model_dict.items():
        p = f"{path}/{prefix}__{k}_{current_time}.pth"
        torch.save(v, p)
        # print(f"Saved: {p}")

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad