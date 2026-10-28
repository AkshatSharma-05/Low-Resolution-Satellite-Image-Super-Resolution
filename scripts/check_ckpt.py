import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.generator import Generator

for name in ["generator_pretrain_best.pth", "generator_pretrain_latest.pth"]:
    p = PROJECT_ROOT / "checkpoints" / name
    if not p.exists():
        print(f"Missing: {name}")
        continue
    ckpt = torch.load(p, map_location="cpu")
    print(f"=== {name} ===")
    ep = ckpt.get("epoch", -1)
    print(f"  Epoch       : {ep + 1 if isinstance(ep, int) else ep}")
    print(f"  Train Loss  : {ckpt.get('train_loss', 'N/A')}")
    print(f"  Val Loss    : {ckpt.get('val_loss', 'N/A')}")
    print(f"  Val PSNR    : {ckpt.get('val_psnr', 'N/A')}")
    print(f"  Val SSIM    : {ckpt.get('val_ssim', 'N/A')}")
    print(f"  Bicubic PSNR: {ckpt.get('bicubic_psnr', 'N/A')}")
    print(f"  Bicubic SSIM: {ckpt.get('bicubic_ssim', 'N/A')}")

# Verify architecture compatibility
netG = Generator()
best_ckpt = torch.load(PROJECT_ROOT / "checkpoints" / "generator_pretrain_best.pth", map_location="cpu")
res = netG.load_state_dict(best_ckpt["model_state_dict"], strict=True)
print("\nGenerator load_state_dict strict=True result:", res)
print("Parameters count:", sum(p.numel() for p in netG.parameters()))
