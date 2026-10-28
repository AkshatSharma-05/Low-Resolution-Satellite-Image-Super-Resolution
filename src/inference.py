"""
Inference utilities for Satellite Image Super-Resolution.
Provides image preprocessing, validation, model execution, and metric assessment
used by both the evaluation suite and the Streamlit web application.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from src import config
from src.baseline import bicubic_upscale
from src.dataset import denormalize_hr, tensor_to_numpy_image
from src.metrics import calculate_psnr, calculate_ssim, to_numpy_image
from src.models.generator import Generator


def validate_and_preprocess_image(
    img: Image.Image,
    lr_size: int = config.LR_SIZE,
    hr_size: int = config.HR_SIZE,
) -> Tuple[Image.Image, Optional[Image.Image], str]:
    """
    Validate and preprocess an input PIL Image for 4x super-resolution.
    
    Rules:
    - Converts any mode (RGBA, Grayscale, CMYK, P) to RGB.
    - If input is exactly (lr_size, lr_size): treated directly as LR input without HR reference.
    - If input is exactly (hr_size, hr_size): treated as ground truth HR, downsampled to LR via BICUBIC.
    - If input is larger/different: resized to (hr_size, hr_size) as reference HR and degraded to (lr_size, lr_size).
    
    Args:
        img: Input PIL Image.
        lr_size: Target LR dimension (default 64).
        hr_size: Target HR dimension (default 256).
        
    Returns:
        lr_img: (lr_size, lr_size) RGB PIL Image.
        hr_img: (hr_size, hr_size) RGB PIL Image if ground truth / reference is available, else None.
        status_msg: Description of preprocessing applied.
    """
    if not isinstance(img, Image.Image):
        raise TypeError(f"Expected PIL Image, got {type(img)}")
        
    # Convert to RGB mode
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    w, h = img.size
    
    if (w, h) == (lr_size, lr_size):
        return img, None, f"Input matches low-resolution format ({lr_size}×{lr_size}). No ground-truth reference."
        
    if (w, h) == (hr_size, hr_size):
        lr_img = img.resize((lr_size, lr_size), resample=Image.Resampling.BICUBIC)
        return lr_img, img, f"Input is high-resolution ({hr_size}×{hr_size}). Generated 4× degraded LR ({lr_size}×{lr_size}) for evaluation."
        
    # Other dimensions: Resize to hr_size, then degrade to lr_size
    hr_ref = img.resize((hr_size, hr_size), resample=Image.Resampling.BICUBIC)
    lr_img = hr_ref.resize((lr_size, lr_size), resample=Image.Resampling.BICUBIC)
    return lr_img, hr_ref, f"Input ({w}×{h}) resized to reference HR ({hr_size}×{hr_size}) and degraded to LR ({lr_size}×{lr_size})."


def pil_to_lr_tensor(lr_img: Image.Image) -> torch.Tensor:
    """
    Convert LR PIL Image to normalized PyTorch tensor (1, 3, H, W) in [0.0, 1.0].
    """
    if lr_img.mode != "RGB":
        lr_img = lr_img.convert("RGB")
    tensor = TF.to_tensor(lr_img).unsqueeze(0)  # Shape (1, 3, H, W) in [0, 1]
    return tensor


def tensor_to_pil(tensor: torch.Tensor, is_hr: bool = False) -> Image.Image:
    """
    Convert PyTorch tensor to PIL Image.
    """
    np_arr = tensor_to_numpy_image(tensor, is_hr=is_hr)
    return Image.fromarray(np_arr)


def run_super_resolution(
    lr_img: Image.Image,
    model_pretrain: Generator,
    model_srgan: Generator,
    device: torch.device,
) -> Dict[str, Image.Image]:
    """
    Run 4x super-resolution across Bicubic, Pretrained SRResNet, and SRGAN.
    
    Args:
        lr_img: Low-resolution PIL Image (64x64 RGB).
        model_pretrain: Pretrained SRResNet model.
        model_srgan: Trained SRGAN model.
        device: PyTorch device (CPU or CUDA).
        
    Returns:
        Dictionary mapping model names to super-resolved PIL Images (256x256 RGB).
    """
    # 1. Bicubic Baseline
    lr_tensor = TF.to_tensor(lr_img)  # (3, 64, 64) in [0, 1]
    sr_bicubic_tensor = bicubic_upscale(lr_tensor, target_size=config.HR_SIZE)
    bicubic_pil = tensor_to_pil(sr_bicubic_tensor, is_hr=False)
    
    # 2. Pretrained SRResNet & SRGAN
    lr_batch = lr_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        sr_pre_tanh = model_pretrain(lr_batch).squeeze(0).cpu()
        sr_pre_tensor = denormalize_hr(sr_pre_tanh)
        pretrain_pil = tensor_to_pil(sr_pre_tensor, is_hr=False)
        
        sr_srgan_tanh = model_srgan(lr_batch).squeeze(0).cpu()
        sr_srgan_tensor = denormalize_hr(sr_srgan_tanh)
        srgan_pil = tensor_to_pil(sr_srgan_tensor, is_hr=False)
        
    return {
        "bicubic": bicubic_pil,
        "pretrain": pretrain_pil,
        "srgan": srgan_pil,
    }


def compute_image_metrics(
    target_img: Image.Image,
    pred_img: Image.Image,
) -> Dict[str, float]:
    """
    Compute PSNR, SSIM, and MSE between target and predicted PIL Images.
    """
    target_np = np.asarray(target_img.convert("RGB")).astype(np.float32) / 255.0
    pred_np = np.asarray(pred_img.convert("RGB")).astype(np.float32) / 255.0
    
    psnr_val = calculate_psnr(target_np, pred_np)
    ssim_val = calculate_ssim(target_np, pred_np)
    mse_val = float(np.mean((target_np - pred_np) ** 2))
    
    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "mse": mse_val,
    }
