"""
Satellite Image Dataset Pipeline for 4x Super-Resolution (64x64 -> 256x256).
Supports UC Merced Land Use TIFF images with class-stratified train/val/test splits.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from src import config


def discover_images(raw_dir: Union[str, Path] = config.RAW_IMAGES_DIR) -> Dict[str, List[Path]]:
    """
    Recursively discover all image files organized by category folder.
    
    Args:
        raw_dir: Directory containing category subfolders with images.
        
    Returns:
        Dict mapping category name to a sorted list of image file Paths.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw images directory does not exist: {raw_path}")
        
    supported_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    categories: Dict[str, List[Path]] = {}
    
    for category_dir in sorted(raw_path.iterdir()):
        if category_dir.is_dir():
            images = [
                p for p in sorted(category_dir.iterdir())
                if p.is_file() and p.suffix.lower() in supported_exts
            ]
            if images:
                categories[category_dir.name] = images
                
    if not categories:
        raise ValueError(f"No valid images found in category subdirectories under: {raw_path}")
        
    return categories


def create_stratified_splits(
    categories: Optional[Dict[str, List[Path]]] = None,
    splits_dir: Union[str, Path] = config.SPLITS_DIR,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
    random_seed: int = config.RANDOM_SEED,
    force_recreate: bool = False,
) -> Dict[str, List[str]]:
    """
    Create or load a reproducible, class-stratified train/val/test split.
    Each image belongs to exactly one split.
    
    Args:
        categories: Category-to-images mapping. If None, discovered automatically.
        splits_dir: Directory where the split manifest will be stored.
        train_ratio: Fraction for training (default 0.80).
        val_ratio: Fraction for validation (default 0.10).
        test_ratio: Fraction for testing (default 0.10).
        random_seed: Seed for reproducible random shuffling.
        force_recreate: If True, recreate the manifest even if it already exists.
        
    Returns:
        Dict with keys 'train', 'val', 'test', each containing relative file paths.
    """
    splits_path = Path(splits_dir)
    splits_path.mkdir(parents=True, exist_ok=True)
    manifest_file = splits_path / "splits_manifest.json"
    
    if manifest_file.exists() and not force_recreate:
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return manifest["splits"]

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios train + val + test must sum to 1.0")

    if categories is None:
        categories = discover_images()

    train_files: List[str] = []
    val_files: List[str] = []
    test_files: List[str] = []
    
    rng = random.Random(random_seed)
    raw_root = Path(config.RAW_IMAGES_DIR)

    for cat_name, img_paths in sorted(categories.items()):
        # Work on sorted list for determinism, then shuffle with fixed seed
        shuffled = list(img_paths)
        rng.shuffle(shuffled)
        
        n_total = len(shuffled)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        # Ensure test gets the remainder to avoid rounding issues
        n_test = n_total - (n_train + n_val)
        
        cat_train = shuffled[:n_train]
        cat_val = shuffled[n_train:n_train + n_val]
        cat_test = shuffled[n_train + n_val:]
        
        train_files.extend(str(p.relative_to(raw_root).as_posix()) for p in cat_train)
        val_files.extend(str(p.relative_to(raw_root).as_posix()) for p in cat_val)
        test_files.extend(str(p.relative_to(raw_root).as_posix()) for p in cat_test)

    # Sort manifest entries for readability and consistent diffs
    train_files.sort()
    val_files.sort()
    test_files.sort()

    manifest_data = {
        "metadata": {
            "random_seed": random_seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "total_images": len(train_files) + len(val_files) + len(test_files),
            "counts": {
                "train": len(train_files),
                "val": len(val_files),
                "test": len(test_files),
            },
            "num_classes": len(categories),
        },
        "splits": {
            "train": train_files,
            "val": val_files,
            "test": test_files,
        },
    }

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_data["splits"]


def degrade_to_lr(hr_img: Image.Image, lr_size: int = config.LR_SIZE) -> Image.Image:
    """
    Degrade a High-Resolution (256x256) PIL Image to Low-Resolution (64x64)
    using bicubic interpolation (PIL.Image.Resampling.BICUBIC).
    
    This is the standard, documented downsampling method used in super-resolution
    benchmarks (e.g. SRGAN, RCAN, EDSR).
    
    Args:
        hr_img: Input high-resolution PIL Image.
        lr_size: Target low-resolution square dimension (default 64).
        
    Returns:
        Downsampled PIL Image of dimension (lr_size, lr_size).
    """
    return hr_img.resize((lr_size, lr_size), resample=Image.Resampling.BICUBIC)


def denormalize_hr(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert normalized HR tensor from [-1, 1] range to [0, 1] range.
    Clamps values to [0.0, 1.0].
    """
    return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)


def denormalize_lr(tensor: torch.Tensor) -> torch.Tensor:
    """
    Return LR tensor clamped to [0, 1].
    """
    return torch.clamp(tensor, 0.0, 1.0)


def tensor_to_numpy_image(tensor: torch.Tensor, is_hr: bool = False) -> np.ndarray:
    """
    Convert a (C, H, W) PyTorch image tensor to (H, W, C) uint8 numpy array [0, 255].
    
    Args:
        tensor: (C, H, W) or (1, C, H, W) PyTorch tensor.
        is_hr: If True, denormalizes from [-1, 1] to [0, 1]. If False, assumes [0, 1].
        
    Returns:
        uint8 numpy array of shape (H, W, C).
    """
    t = tensor.detach().cpu().squeeze(0) if tensor.dim() == 4 else tensor.detach().cpu()
    if is_hr:
        t = denormalize_hr(t)
    else:
        t = denormalize_lr(t)
    t = t.permute(1, 2, 0).numpy()
    img_uint8 = np.clip(t * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return img_uint8


class SatelliteSRDataset(Dataset):
    """
    PyTorch Dataset for Satellite Image Super-Resolution.
    
    Returns:
        lr_tensor: (3, 64, 64) float32 in [0.0, 1.0] range.
        hr_tensor: (3, 256, 256) float32 in [-1.0, 1.0] range (standard for SRGAN).
        info_dict: Metadata including relative path, category, and original dimensions.
    """
    def __init__(
        self,
        split: str = "train",
        splits_dir: Union[str, Path] = config.SPLITS_DIR,
        raw_images_dir: Union[str, Path] = config.RAW_IMAGES_DIR,
        hr_size: int = config.HR_SIZE,
        lr_size: int = config.LR_SIZE,
        augment: bool = False,
    ):
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split '{split}'. Expected 'train', 'val', or 'test'.")

        self.split = split
        self.raw_images_dir = Path(raw_images_dir)
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.augment = augment and (split == "train")

        splits = create_stratified_splits(splits_dir=splits_dir)
        self.relative_paths = splits[split]

    def __len__(self) -> int:
        return len(self.relative_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, str]]:
        rel_path = self.relative_paths[idx]
        full_path = self.raw_images_dir / rel_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")

        # Open TIFF image and ensure 3-channel RGB mode
        with Image.open(full_path) as img:
            hr_pil = img.convert("RGB")

        # Verify HR dimension or resize to standard hr_size if needed
        if hr_pil.size != (self.hr_size, self.hr_size):
            hr_pil = hr_pil.resize((self.hr_size, self.hr_size), resample=Image.Resampling.BICUBIC)

        # Apply data augmentations (flips / 90-degree rotations) for training
        if self.augment:
            if random.random() > 0.5:
                hr_pil = TF.hflip(hr_pil)
            if random.random() > 0.5:
                hr_pil = TF.vflip(hr_pil)
            rotations = [0, 90, 180, 270]
            rot = random.choice(rotations)
            if rot > 0:
                hr_pil = TF.rotate(hr_pil, rot)

        # Generate low-resolution image via bicubic degradation
        lr_pil = degrade_to_lr(hr_pil, lr_size=self.lr_size)

        # Convert to float32 Tensors (C, H, W) in [0.0, 1.0] range
        lr_tensor = TF.to_tensor(lr_pil)  # Shape (3, 64, 64), Range [0, 1]
        hr_tensor = TF.to_tensor(hr_pil)  # Shape (3, 256, 256), Range [0, 1]

        # Normalize HR tensor to [-1.0, 1.0] for SRGAN Generator/Tanh output
        hr_normalized = hr_tensor * 2.0 - 1.0

        # Category from folder structure (rel_path is <category>/<filename>)
        category = Path(rel_path).parent.name

        info = {
            "rel_path": rel_path,
            "category": category,
            "lr_shape": f"{lr_tensor.shape}",
            "hr_shape": f"{hr_normalized.shape}",
        }

        return lr_tensor, hr_normalized, info


def get_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    pin_memory: bool = config.PIN_MEMORY,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience helper to create train, val, and test DataLoaders.
    """
    train_ds = SatelliteSRDataset(split="train", augment=True)
    val_ds = SatelliteSRDataset(split="val", augment=False)
    test_ds = SatelliteSRDataset(split="test", augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing image discovery and split creation...")
    cats = discover_images()
    print(f"Discovered {len(cats)} categories, total {sum(len(v) for v in cats.values())} images.")
    splits = create_stratified_splits(categories=cats)
    for k, v in splits.items():
        print(f"  Split '{k}': {len(v)} samples")

    print("\nTesting Dataset loading...")
    ds = SatelliteSRDataset(split="test")
    lr, hr, info = ds[0]
    print(f"Sample 0 Info     : {info}")
    print(f"LR Tensor Shape   : {lr.shape}, dtype: {lr.dtype}, Range: [{lr.min():.3f}, {lr.max():.3f}]")
    print(f"HR Tensor Shape   : {hr.shape}, dtype: {hr.dtype}, Range: [{hr.min():.3f}, {hr.max():.3f}]")
    hr_denorm = denormalize_hr(hr)
    print(f"Denorm HR Range   : [{hr_denorm.min():.3f}, {hr_denorm.max():.3f}]")
