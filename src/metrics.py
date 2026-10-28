"""
Image Quality Assessment Metrics for Super-Resolution:
PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index Measure).
"""

from typing import Union
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def to_numpy_image(img: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Convert PyTorch tensor (C, H, W) or (B, C, H, W) to NumPy (H, W, C) in [0.0, 1.0] range.
    """
    if isinstance(img, torch.Tensor):
        t = img.detach().cpu()
        if t.dim() == 4:
            t = t.squeeze(0)
        if t.shape[0] in (1, 3):  # (C, H, W)
            t = t.permute(1, 2, 0)
        arr = t.numpy()
    else:
        arr = np.asarray(img)
        
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def calculate_psnr(
    target: Union[torch.Tensor, np.ndarray],
    pred: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0,
) -> float:
    """
    Calculate PSNR between target and prediction images in [0, 1] range.
    
    Args:
        target: Ground truth image (C, H, W) or (H, W, C).
        pred: Predicted / super-resolved image.
        data_range: Dynamic range of input images (default 1.0).
        
    Returns:
        PSNR value in decibels (dB).
    """
    target_np = to_numpy_image(target)
    pred_np = to_numpy_image(pred)
    
    mse = np.mean((target_np - pred_np) ** 2)
    if mse == 0:
        return float("inf")
        
    return float(peak_signal_noise_ratio(target_np, pred_np, data_range=data_range))


def calculate_ssim(
    target: Union[torch.Tensor, np.ndarray],
    pred: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0,
) -> float:
    """
    Calculate mean SSIM across all channels between target and prediction in [0, 1] range.
    
    Args:
        target: Ground truth image (C, H, W) or (H, W, C).
        pred: Predicted / super-resolved image.
        data_range: Dynamic range of input images (default 1.0).
        
    Returns:
        SSIM score in [-1.0, 1.0] (typically [0, 1]).
    """
    target_np = to_numpy_image(target)
    pred_np = to_numpy_image(pred)
    
    # Determine channel axis
    channel_axis = -1 if target_np.ndim == 3 and target_np.shape[-1] in (1, 3) else None
    
    return float(
        structural_similarity(
            target_np,
            pred_np,
            data_range=data_range,
            channel_axis=channel_axis,
        )
    )
