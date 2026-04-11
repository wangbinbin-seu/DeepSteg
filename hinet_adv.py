import torch
import torch.nn as nn
import torch.optim
import numpy as np
import logging
import os
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision import models

from utils.terminal import MetricMonitor
from utils.logger import logger_info
from utils.image import IWT, DWT, gauss_noise, calculate_psnr, quantization, calculate_ssim, calculate_mae, calculate_rmse
from data.dataset import load_dataset
from models.HiNet import Model as HiNetModel, init_model as init_hinet
from models.SRNet import Model as SRNetModel
from models.XuNet import XuNet as XuNetModel
from utils.dirs import mkdirs
from utils.model import load_model
from utils.runtime import get_device, wrap_model_for_cuda
import config_steg as c


USE_SRNET = True  # True: SRNet, False: XuNet

if c.hinet_device_ids:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(v) for v in c.hinet_device_ids])

device = get_device()
model_save_path = os.path.join(c.model_dir, 'hinet_adv')
mkdirs(model_save_path)
train_data_dir = os.path.join(c.data_dir, c.data_name_train, 'train')
test_data_dir = os.path.join(c.data_dir, c.data_name_test, 'test')

mkdirs('results')
logger_name = 'hinet_adv_trained_ON_' + c.data_name_train
logger_info(logger_name, log_path=os.path.join('results', logger_name + '.log'))
logger = logging.getLogger(logger_name)
logger.info('#' * 50)
logger.info('model: hinet_adv')
logger.info('train data: {:s}'.format(c.data_name_train))
logger.info('test data: {:s}'.format(c.data_name_test))
logger.info('mode: {:s}'.format(c.mode))


train_loader, test_loader = load_dataset(
    train_data_dir,
    test_data_dir,
    c.hinet_batch_size_train,
    c.hinet_batch_size_test,
    num_workers_train=getattr(c, "num_workers_train", 8),
    num_workers_test=getattr(c, "num_workers_test", 2),
)

dwt = DWT()
iwt = IWT()


def high_freq_features(img):
    comps = dwt(img)
    return comps[:, c.channels_in:, :, :]


class VGGPerceptual(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features[:16]
        self.backbone = vgg.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.backbone(x)


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


hinet = HiNetModel()
selected_device_ids = list(range(len(c.hinet_device_ids))) if c.hinet_device_ids else []
hinet = wrap_model_for_cuda(hinet, device, selected_device_ids if c.use_data_parallel else [])

if USE_SRNET:
    steganalyzer = SRNetModel().to(device)
else:
    steganalyzer = XuNetModel().to(device)

vgg_perc = VGGPerceptual().to(device)

steg_optimizer = torch.optim.Adam(steganalyzer.parameters(), lr=1e-4, weight_decay=1e-5)
recon_loss = nn.L1Loss().to(device)
clf_criterion = nn.CrossEntropyLoss().to(device)

lambda_S = 1.0
lambda_R = 1.0
lambda_freq = 0.1
lambda_perc = 0.05
lambda_adv = 0.1


def hinet_forward(batch):
    secret = batch[batch.shape[0] // 2:]
    cover = batch[:batch.shape[0] // 2]
    cover_input = dwt(cover)
    secret_input = dwt(secret)
    input_img = torch.cat((cover_input, secret_input), 1)

    output = hinet(input_img)
    output_stego = output.narrow(1, 0, 4 * c.channels_in)
    stego = iwt(output_stego)
    stego = quantization(stego)

    output_stego_dwt = dwt(stego)
    output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
    output_z = gauss_noise(output_z.shape, device=device)

    output_rev = torch.cat((output_stego_dwt, output_z), 1)
    output_image = hinet(output_rev, rev=True)
    secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
    secret_rev = iwt(secret_rev)
    secret_rev = quantization(secret_rev)

    return cover, stego, secret, secret_rev


def steganalyzer_logits(covers, stegos):
    inputs = torch.cat([covers, stegos], dim=0)
    labels = torch.cat([
        torch.zeros(covers.shape[0], dtype=torch.long, device=device),
        torch.ones(stegos.shape[0], dtype=torch.long, device=device),
    ], dim=0)
    logits = steganalyzer(inputs)
    return logits, labels


if c.mode == 'test':
    raise NotImplementedError("Please use original hinet.py for testing; hinet_adv.py focuses on advanced training experiments.")


init_hinet(hinet)
if c.use_data_parallel and selected_device_ids:
    hinet = torch.nn.DataParallel(hinet, device_ids=selected_device_ids)

optimizer = torch.optim.Adam(hinet.parameters(), lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, c.weight_step, gamma=c.gamma)


for epoch in range(c.epochs):
    epoch += 1
    metric_monitor = MetricMonitor(float_precision=4)

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{c.epochs} - HiNet_adv"):
        batch = batch.to(device)

        # ----- Train steganalyzer -----
        hinet.eval()
        steganalyzer.train()
        set_requires_grad(steganalyzer, True)

        with torch.no_grad():
            cover, stego, _, _ = hinet_forward(batch)

        logits_D, labels_D = steganalyzer_logits(cover.detach(), stego.detach())
        loss_D = clf_criterion(logits_D, labels_D)
        steg_optimizer.zero_grad()
        loss_D.backward()
        steg_optimizer.step()

        # ----- Train HiNet -----
        hinet.train()
        steganalyzer.eval()
        set_requires_grad(steganalyzer, False)

        cover, stego, secret, secret_rev = hinet_forward(batch)

        logits_G, labels = steganalyzer_logits(cover, stego)
        adv_target = torch.zeros_like(labels)
        loss_adv = clf_criterion(logits_G, adv_target)

        loss_cover = recon_loss(stego, cover)
        loss_secret = recon_loss(secret_rev, secret)
        loss_freq = recon_loss(high_freq_features(stego), high_freq_features(cover))
        loss_perc = recon_loss(vgg_perc(stego), vgg_perc(cover))

        total_loss = (
            lambda_S * loss_cover
            + lambda_R * loss_secret
            + lambda_freq * loss_freq
            + lambda_perc * loss_perc
            + lambda_adv * loss_adv
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        metric_monitor.update("L_cov", loss_cover.item())
        metric_monitor.update("L_sec", loss_secret.item())
        metric_monitor.update("L_freq", loss_freq.item())
        metric_monitor.update("L_perc", loss_perc.item())
        metric_monitor.update("L_adv", loss_adv.item())
        metric_monitor.update("L_D", loss_D.item())

    scheduler.step()
    logger.info(
        f"Epoch {epoch}/{c.epochs} | L_cov={metric_monitor.metrics['L_cov']['avg']:.4f}"
        f", L_sec={metric_monitor.metrics['L_sec']['avg']:.4f}, L_freq={metric_monitor.metrics['L_freq']['avg']:.4f}"
        f", L_perc={metric_monitor.metrics['L_perc']['avg']:.4f}, L_adv={metric_monitor.metrics['L_adv']['avg']:.4f}, L_D={metric_monitor.metrics['L_D']['avg']:.4f}"
    )

    if epoch % c.hinet_save_freq == 0:
        torch.save(hinet.state_dict(), os.path.join(model_save_path, f"checkpoint_{epoch:04d}.pt"))
