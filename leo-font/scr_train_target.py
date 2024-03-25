import torch
import torch.nn.functional as F
import itertools
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SCRDataset
from models import Discriminator, ContentEncoder, StyleEncoder, Decoder, SCR
from losses import DisLoss
from accelerate import Accelerator
from configs import SCRConfig

configs = SCRConfig()
batch_size = configs.batch_size
lr = configs.lr
g_ch = configs.G_ch
n_embedding = configs.n_embedding
dataset_path = configs.dataset_path
scr_dataset_path = configs.scr_dataset_path
scr_model_path = configs.scr_model_path
n_epoch = configs.n_epoch
scr_coef = configs.scr_coef

accelerator = Accelerator()
dataset = SCRDataset(path=dataset_path, scr_fd=scr_dataset_path)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
discriminator = Discriminator()
content_encoder = ContentEncoder(G_ch=g_ch)
style_encoder = StyleEncoder(G_ch=g_ch)
decoder = Decoder(G_ch=g_ch, nEmbedding=n_embedding)

scr = SCR() # image_size: 96
scr.load_state_dict(torch.load(scr_model_path))
scr.requires_grad_(False)

optimizer_G = torch.optim.Adam(itertools.chain(
    content_encoder.parameters(),
    style_encoder.parameters(),
    decoder.parameters()), lr=lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

criterionD = DisLoss()

dataloader, content_encoder, style_encoder, decoder, discriminator, optimizer_D, optimizer_G = accelerator.prepare(
    dataloader, content_encoder, style_encoder, decoder, discriminator, optimizer_D, optimizer_G)

scr = scr.to(accelerator.device)

for epoch in tqdm(range(n_epoch)):
    for data in dataloader:
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()

        cont, residulte_features = content_encoder(data['content'])
        style_emb, style_fc, residual_features_style = style_encoder(data['style'])
        img_print2write = decoder(cont, residulte_features, style_emb, style_fc, residual_features_style)

        real_out = discriminator(data['target'])
        fake_out = discriminator(img_print2write)

        sample_style_embeddings, pos_style_embeddings, neg_style_embeddings = scr(
            F.interpolate(img_print2write, size=(96, 96), mode='bilinear', align_corners=False),
            data['target'],
            data['negative'],
            nce_layers='0,1,2,3')
        sc_loss = scr.calculate_nce_loss(
            sample_s=sample_style_embeddings,
            pos_s=pos_style_embeddings,
            neg_s=neg_style_embeddings)

        loss_D = criterionD(real_out, True) + criterionD(fake_out, False)
        loss_G = criterionD(fake_out, True) + scr_coef * sc_loss

        accelerator.backward(loss_D)
        optimizer_D.step()

        accelerator.backward(loss_G)
        optimizer_G.step()
