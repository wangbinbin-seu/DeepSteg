import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from utils.image import DWT, IWT, quantization
from models.HiNet import Model as HiNetModel, init_model as init_hinet
from utils.runtime import get_device


# 使用 mini DIV2K 子集做 CPU 功能性验证（路径基于项目根目录）
ROOT_DIR = Path(__file__).resolve().parent  # DeepSteg 根目录
DATA_ROOT = ROOT_DIR / "data/steganography/div2k_mini"
TRAIN_DIR = DATA_ROOT / "train"


def load_images(n=10, size=256):
    """从 TRAIN_DIR 读前 n 张图片，缩放到 size×size，转为 [0,1] tensor。"""
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    paths = sorted(TRAIN_DIR.glob("*.png"))[:n]
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(tfm(img))
    return torch.stack(imgs, dim=0)  # [N,3,H,W]


def hinet_step(hinet, dwt, iwt, batch):
    """执行一次 HiNet forward+backward，用简单 L1 重建损失。"""
    device = next(hinet.parameters()).device
    batch = batch.to(device)
    # 简单起见：前半当 cover，后半当 secret；若数量为奇数，最后一张重复
    if batch.shape[0] % 2 == 1:
        batch = torch.cat([batch, batch[-1:]], dim=0)
    n = batch.shape[0] // 2
    cover = batch[:n]
    secret = batch[n:]

    cover_dwt = dwt(cover)
    secret_dwt = dwt(secret)
    x = torch.cat([cover_dwt, secret_dwt], dim=1)

    # forward
    y = hinet(x)
    y_stego = y[:, :4 * 3, :, :]  # 4*channels_in
    stego = iwt(y_stego)
    stego = quantization(stego)

    # backward (只做一次反向，用 L1)
    loss_fn = nn.L1Loss()
    loss = loss_fn(stego, cover)
    loss.backward()
    return loss.item()


def main():
    device = get_device()
    print("Using device:", device)

    if not TRAIN_DIR.exists():
        raise SystemExit(f"TRAIN_DIR not found: {TRAIN_DIR}")

    imgs = load_images(n=10, size=256)
    print(f"Loaded {imgs.shape[0]} images from {TRAIN_DIR}")

    hinet = HiNetModel().to(device)
    init_hinet(hinet)
    dwt = DWT().to(device)
    iwt = IWT().to(device)

    optimizer = torch.optim.Adam(hinet.parameters(), lr=1e-4)

    hinet.train()
    optimizer.zero_grad()
    loss_val = hinet_step(hinet, dwt, iwt, imgs)
    optimizer.step()
    print(f"One CPU sanity step done. L1 cover loss = {loss_val:.4f}")


if __name__ == "__main__":
    main()
