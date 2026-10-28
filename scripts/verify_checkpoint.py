from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.models.generator import Generator
from src.train_pretrain import load_checkpoint
from src.dataset import SatelliteSRDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)

ckpt_path = PROJECT_ROOT / "checkpoints" / "generator_pretrain_latest.pth"
info = load_checkpoint(ckpt_path, netG, device=device)

print(f"Checkpoint successfully loaded from epoch {info['epoch'] + 1}!")
print(f"Recorded Validation PSNR: {info['val_psnr']:.2f} dB, SSIM: {info['val_ssim']:.4f}")

# Inference test
ds = SatelliteSRDataset(split="val")
lr, hr, meta = ds[0]
netG.eval()
with torch.no_grad():
    sr = netG(lr.unsqueeze(0).to(device))

print("Inference verification successful:")
print(f"  Input LR shape : {tuple(lr.shape)}, value range [{lr.min():.3f}, {lr.max():.3f}]")
print(f"  Output SR shape: {tuple(sr.shape)}, value range [{sr.min():.3f}, {sr.max():.3f}]")
