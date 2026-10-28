import torch
from pathlib import Path

ckpt_dir = Path("checkpoints")
for name in [
    "generator_pretrain_best.pth",
    "generator_pretrain_latest.pth",
    "srgan_generator_best.pth",
    "srgan_generator_latest.pth",
    "srgan_discriminator_best.pth",
    "srgan_discriminator_latest.pth",
]:
    p = ckpt_dir / name
    if not p.exists():
        print(f"{name}: Not found")
        continue
    ckpt = torch.load(p, map_location="cpu")
    print(f"=== {name} ===")
    keys = list(ckpt.keys())
    print(f"  Keys: {keys}")
    for k in ["epoch", "val_psnr", "val_ssim", "val_loss", "train_loss", "config"]:
        if k in ckpt:
            val = ckpt[k]
            if k == "epoch":
                print(f"  {k}: {val} (Epoch {val+1})")
            else:
                print(f"  {k}: {val}")
