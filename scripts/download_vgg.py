import os
from pathlib import Path
import sys
import torch
import torchvision.models as models

print("Downloading official pretrained VGG-19 weights via Torchvision API...")
weights = models.VGG19_Weights.IMAGENET1K_V1
vgg19 = models.vgg19(weights=weights)
vgg19.eval()
print("VGG-19 model loaded successfully into memory!")

# Locate cache file
torch_home = Path(torch.hub.get_dir()) / "checkpoints"
print(f"Torch Hub Checkpoints Directory: {torch_home}")

cached_files = list(torch_home.glob("*vgg19*"))
if cached_files:
    for f in cached_files:
        size_mb = f.stat().st_size / (1024 ** 2)
        print(f"Cached file: {f.name} | Size: {size_mb:.2f} MB ({f.stat().st_size:,} bytes) | Path: {f.resolve()}")
else:
    print("Warning: Could not find vgg19 file in torch checkpoints directory.")
