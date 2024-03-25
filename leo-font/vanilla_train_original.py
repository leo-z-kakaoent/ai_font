import torch
import itertools
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FontDataset
from models import Discriminator, ContentEncoder, StyleEncoder, Decoder
from losses import DisLoss
from accelerate import Accelerator

lr = 0.0002

accelerator = Accelerator()
dataset = FontDataset(path="../../data/raw/")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
discriminator = Discriminator()
content_encoder = ContentEncoder(G_ch=64)
style_encoder = StyleEncoder(G_ch=64)
decoder = Decoder(G_ch=64, nEmbedding=1024)

optimizer_G = torch.optim.Adam(itertools.chain(
    content_encoder.parameters(),
    style_encoder.parameters(),
    decoder.parameters()), lr=lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

criterionD = DisLoss()

dataloader, content_encoder, style_encoder, decoder, discriminator, optimizer_D, optimizer_G = accelerator.prepare(
    dataloader, content_encoder, style_encoder, decoder, discriminator, optimizer_D, optimizer_G)

for epoch in tqdm(range(100)):
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
