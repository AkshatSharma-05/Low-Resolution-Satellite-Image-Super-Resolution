"""
Configuration file for Satellite Image Super-Resolution using SRGAN.
Optimized for NVIDIA RTX 3050 (6GB VRAM) and 16GB RAM on Windows 11.
"""

from pathlib import Path
import os
import torch

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_IMAGES_DIR = DATA_DIR / "raw" / "Images"
SPLITS_DIR = DATA_DIR / "splits"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
COMPARISONS_DIR = RESULTS_DIR / "comparisons"

# Image & Super-Resolution parameters
HR_SIZE = 256
LR_SIZE = 64
UPSCALE_FACTOR = 4
CHANNELS = 3

# Training Parameters - Configurable via environment variables or CLI overrides
# Defaults to batch size 4 as requested
BATCH_SIZE = int(os.getenv("SRGAN_BATCH_SIZE", "4"))
NUM_WORKERS = int(os.getenv("SRGAN_NUM_WORKERS", "2"))
PIN_MEMORY = True

# Hardware acceleration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_AMP = True  # Automatic Mixed Precision (torch.cuda.amp) to conserve VRAM

# Phase 1: Generator Pretraining (SRResNet)
PRETRAIN_EPOCHS = int(os.getenv("SRGAN_PRETRAIN_EPOCHS", "30"))
PRETRAIN_LR = float(os.getenv("SRGAN_PRETRAIN_LR", "1e-4"))

# Phase 2: Adversarial Training (SRGAN)
SRGAN_EPOCHS = int(os.getenv("SRGAN_EPOCHS", "50"))
GENERATOR_LR = float(os.getenv("SRGAN_GEN_LR", "1e-4"))
DISCRIMINATOR_LR = float(os.getenv("SRGAN_DISC_LR", "2e-5"))
DISCRIMINATOR_LABEL_SMOOTHING = float(os.getenv("SRGAN_LABEL_SMOOTHING", "0.1"))
ADVERSARIAL_WEIGHT = 1e-3
CONTENT_VGG_WEIGHT = float(os.getenv("SRGAN_VGG_WEIGHT", "0.1"))
DISCRIMINATOR_UPDATE_INTERVAL = int(os.getenv("SRGAN_DISC_INTERVAL", "2"))

# Data split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42

def get_config():
    """Return dictionary representation of the configuration."""
    return {
        "project_root": str(PROJECT_ROOT),
        "raw_images_dir": str(RAW_IMAGES_DIR),
        "hr_size": HR_SIZE,
        "lr_size": LR_SIZE,
        "upscale_factor": UPSCALE_FACTOR,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "device": str(DEVICE),
        "use_amp": USE_AMP,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "srgan_epochs": SRGAN_EPOCHS,
        "generator_lr": GENERATOR_LR,
        "discriminator_lr": DISCRIMINATOR_LR,
        "discriminator_label_smoothing": DISCRIMINATOR_LABEL_SMOOTHING,
        "adversarial_weight": ADVERSARIAL_WEIGHT,
        "content_vgg_weight": CONTENT_VGG_WEIGHT,
        "discriminator_update_interval": DISCRIMINATOR_UPDATE_INTERVAL,
    }

if __name__ == "__main__":
    import pprint
    pprint.pprint(get_config())
