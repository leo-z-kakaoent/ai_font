import torch
import itertools
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

from dataset import FontDataset
from models import Discriminator, ContentEncoder, StyleEncoder, Decoder
from losses import DisLoss
from accelerate import Accelerator
from configs import DefaultConfig
from utils import save_package, set_requires_grad

configs = DefaultConfig()
lr = configs.lr
dataset_path = configs.dataset_path
batch_size = configs.batch_size
g_ch = configs.G_ch
n_embedding = configs.n_embedding
n_epoch = configs.n_epoch
save_path = configs.save_path

accelerator = Accelerator()
dataset = FontDataset(path=dataset_path)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=6)
discriminator = Discriminator()
content_encoder = ContentEncoder(G_ch=g_ch)
style_encoder = StyleEncoder(G_ch=g_ch)
decoder = Decoder(G_ch=g_ch, nEmbedding=n_embedding)

optimizer_G = torch.optim.Adam(itertools.chain(
    content_encoder.parameters(),
    style_encoder.parameters(),
    decoder.parameters()), lr=lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

criterionD = DisLoss()

dataloader, content_encoder, style_encoder, decoder, discriminator, criterionD, optimizer_D, optimizer_G = accelerator.prepare(
    dataloader, content_encoder, style_encoder, decoder, discriminator, criterionD, optimizer_D, optimizer_G)

pbar = tqdm(total=n_epoch*1000)
c = 0
while c < n_epoch*1000:
    try:
        data = next(loader)
    except:
        loader = iter(dataloader)
        data = next(loader)
        # print("Dataloader Loaded")
    
    cont, residulte_features = content_encoder(data['content'])
    style_emb, style_fc, residual_features_style = style_encoder(data['style'])
    img_print2write = decoder(cont, residulte_features, style_emb, style_fc, residual_features_style)
    fake_out = discriminator(img_print2write)

    loss_G = criterionD(fake_out, True)

    set_requires_grad([content_encoder, style_encoder, decoder], True)
    set_requires_grad([discriminator], False)
    optimizer_G.zero_grad()
    accelerator.backward(loss_G)
    optimizer_G.step()

    cont, residulte_features = content_encoder(data['content'])
    style_emb, style_fc, residual_features_style = style_encoder(data['style'])
    img_print2write = decoder(cont, residulte_features, style_emb, style_fc, residual_features_style)
    fake_out = discriminator(img_print2write)
    real_out = discriminator(data['target'])
    loss_D = criterionD(real_out, True) + criterionD(fake_out, False)

    set_requires_grad([content_encoder, style_encoder, decoder], False)
    set_requires_grad([discriminator], True)
    optimizer_D.zero_grad()
    accelerator.backward(loss_D)
    optimizer_D.step()
    
    pbar.update(1)
    pbar.set_postfix(str=f"loss_G: {loss_G}, loss_D: {loss_D}")

    if c % 100 == 0:
        save_package(
            img_dict={"generated_img": (data['target'][0,:,:,:], img_print2write[0,:,:,:])},
            model_dict={
                "content_encoder": content_encoder.state_dict(),
                "style_encoder": style_encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "discriminator": discriminator.state_dict()
            },
            path=save_path,
            prefix=os.path.basename(__file__).replace(".py", ""),
        )
    c+= 1