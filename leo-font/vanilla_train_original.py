import torch
import itertools
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from dataset import FontDataset
from models import Discriminator, ContentEncoder, StyleEncoder, Decoder
from losses import DisLoss
from accelerate import Accelerator
from configs import DefaultConfig
from utils import save_package

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
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
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

for epoch in tqdm(range(n_epoch)):
    for data in dataloader:
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()

        cont, residulte_features = content_encoder(data['content'])
        style_emb, style_fc, residual_features_style = style_encoder(data['style'])
        img_print2write = decoder(cont, residulte_features, style_emb, style_fc, residual_features_style)

        real_out = discriminator(data['target'])
        fake_out = discriminator(img_print2write)

        loss_D = criterionD(real_out, True) + criterionD(fake_out, False)
        loss_G = criterionD(fake_out, True)

        accelerator.backward(loss_D)
        optimizer_D.step()

        accelerator.backward(loss_G)
        optimizer_G.step()

    save_package(
        img_dict={f"generated{str(i)}": img_print2write[i,:,:,:].detach().cpu().numpy() for i in range(len(img_print2write))},
        model_dict={
            "content_encoder": content_encoder.state_dict(),
            "style_encoder": style_encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "discriminator": discriminator.state_dict()
        },
        path=save_path,
        prefix=os.path.basename(__file__).replace(".py", ""),
    )