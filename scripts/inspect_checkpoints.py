import torch
from pathlib import Path

def inspect(name, pth):
    p = Path(pth)
    if not p.exists():
        print(f"{name}: File not found at {p}")
        return
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    print(f"=== {name} ({p.name}) ===")
    print(f"File size: {p.stat().st_size / (1024*1024):.2f} MB")
    print(f"Epoch: {ckpt.get('epoch')}")
    print(f"Val PSNR: {ckpt.get('val_psnr')}")
    print(f"Val SSIM: {ckpt.get('val_ssim')}")
    print(f"Val Loss: {ckpt.get('val_loss')}")
    keys = list(ckpt.keys())
    print(f"Keys: {keys}")
    if "model_state_dict" in ckpt:
        print(f"Num model params: {len(ckpt['model_state_dict'])}")
    print()

inspect("Pretrained Generator Best", "checkpoints/generator_pretrain_best.pth")
inspect("Pretrained Generator Latest", "checkpoints/generator_pretrain_latest.pth")
inspect("SRGAN Generator Best", "checkpoints/srgan_generator_best.pth")
inspect("SRGAN Generator Latest", "checkpoints/srgan_generator_latest.pth")
inspect("SRGAN Discriminator Best", "checkpoints/srgan_discriminator_best.pth")
inspect("SRGAN Discriminator Latest", "checkpoints/srgan_discriminator_latest.pth")
