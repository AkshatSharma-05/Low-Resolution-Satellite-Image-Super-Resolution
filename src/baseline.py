"""
Bicubic Interpolation Baseline Evaluation for 4x Super-Resolution.
Upscales 64x64 test images to 256x256 and benchmarks PSNR and SSIM against ground truth HR.
Saves metrics summary to results/metrics_summary.json and generates comparison figures.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from collections import defaultdict
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src import config
from src.dataset import SatelliteSRDataset, denormalize_hr, tensor_to_numpy_image
from src.metrics import calculate_psnr, calculate_ssim


def bicubic_upscale(lr_tensor: torch.Tensor, target_size: int = config.HR_SIZE) -> torch.Tensor:
    """
    Upscale an LR tensor (C, H, W) to target_size (target_size, target_size)
    using bicubic interpolation.
    
    Args:
        lr_tensor: Tensor of shape (3, H, W) in [0, 1] range.
        target_size: Output spatial dimension (default 256).
        
    Returns:
        Upscaled tensor of shape (3, target_size, target_size) clamped to [0, 1].
    """
    if lr_tensor.dim() == 3:
        lr_batch = lr_tensor.unsqueeze(0)
    else:
        lr_batch = lr_tensor
        
    sr_bicubic = F.interpolate(
        lr_batch,
        size=(target_size, target_size),
        mode="bicubic",
        align_corners=False,
    )
    sr_bicubic = torch.clamp(sr_bicubic, 0.0, 1.0)
    
    return sr_bicubic.squeeze(0) if lr_tensor.dim() == 3 else sr_bicubic


def evaluate_bicubic(
    test_dataset: SatelliteSRDataset,
    output_json_path: Path = config.RESULTS_DIR / "metrics_summary.json",
) -> Dict[str, Any]:
    """
    Run bicubic baseline evaluation across the test dataset.
    
    Args:
        test_dataset: SatelliteSRDataset instance for the test split.
        output_json_path: Path to save the metrics summary JSON.
        
    Returns:
        Dictionary containing overall and per-class PSNR and SSIM results.
    """
    print(f"\nEvaluating Bicubic Baseline on test set ({len(test_dataset)} images)...")
    
    per_class_psnr = defaultdict(list)
    per_class_ssim = defaultdict(list)
    all_psnr: List[float] = []
    all_ssim: List[float] = []
    
    sample_records = []

    for idx in tqdm(range(len(test_dataset)), desc="Bicubic Evaluation"):
        lr, hr_norm, info = test_dataset[idx]
        hr = denormalize_hr(hr_norm)  # Convert HR to [0, 1] range
        
        # Perform 4x Bicubic upscaling (64x64 -> 256x256)
        bicubic_sr = bicubic_upscale(lr, target_size=config.HR_SIZE)
        
        psnr_val = calculate_psnr(hr, bicubic_sr)
        ssim_val = calculate_ssim(hr, bicubic_sr)
        
        category = info["category"]
        per_class_psnr[category].append(psnr_val)
        per_class_ssim[category].append(ssim_val)
        all_psnr.append(psnr_val)
        all_ssim.append(ssim_val)
        
        sample_records.append({
            "idx": idx,
            "rel_path": info["rel_path"],
            "category": category,
            "psnr": psnr_val,
            "ssim": ssim_val,
        })

    # Compute aggregate statistics
    mean_psnr = float(np.mean(all_psnr))
    std_psnr = float(np.std(all_psnr))
    mean_ssim = float(np.mean(all_ssim))
    std_ssim = float(np.std(all_ssim))

    per_class_summary = {}
    for cat in sorted(per_class_psnr.keys()):
        per_class_summary[cat] = {
            "count": len(per_class_psnr[cat]),
            "mean_psnr": round(float(np.mean(per_class_psnr[cat])), 4),
            "std_psnr": round(float(np.std(per_class_psnr[cat])), 4),
            "mean_ssim": round(float(np.mean(per_class_ssim[cat])), 4),
            "std_ssim": round(float(np.std(per_class_ssim[cat])), 4),
        }

    summary = {
        "method": "Bicubic Interpolation Baseline (4x)",
        "scale_factor": config.UPSCALE_FACTOR,
        "input_resolution": f"{config.LR_SIZE}x{config.LR_SIZE}",
        "target_resolution": f"{config.HR_SIZE}x{config.HR_SIZE}",
        "total_test_samples": len(test_dataset),
        "overall_metrics": {
            "mean_psnr_db": round(mean_psnr, 4),
            "std_psnr_db": round(std_psnr, 4),
            "mean_ssim": round(mean_ssim, 4),
            "std_ssim": round(std_ssim, 4),
        },
        "per_class_metrics": per_class_summary,
    }

    # Save metrics JSON
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n[RESULTS] Bicubic Baseline:")
    print(f"  Mean PSNR : {mean_psnr:.2f} +/- {std_psnr:.2f} dB")
    print(f"  Mean SSIM : {mean_ssim:.4f} +/- {std_ssim:.4f}")
    print(f"  Saved summary to: {output_json_path}")

    return summary


def create_visual_comparison_grid(
    test_dataset: SatelliteSRDataset,
    output_image_path: Path = config.COMPARISONS_DIR / "bicubic_baseline_grid.png",
    sample_categories: Optional[List[str]] = None,
):
    """
    Generate a side-by-side comparison grid: [Low-Res (64x64), Bicubic (256x256), Ground Truth HR (256x256)]
    for representative categories.
    """
    if sample_categories is None:
        sample_categories = ["agricultural", "airplane", "buildings", "freeway", "harbor", "river"]

    # Locate one test sample per target category
    selected_samples = []
    for cat in sample_categories:
        for idx in range(len(test_dataset)):
            rel_path = test_dataset.relative_paths[idx]
            if rel_path.startswith(cat + "/"):
                selected_samples.append((cat, idx))
                break

    n_samples = len(selected_samples)
    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3.2 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, (cat, sample_idx) in enumerate(selected_samples):
        lr, hr_norm, info = test_dataset[sample_idx]
        hr = denormalize_hr(hr_norm)
        bicubic = bicubic_upscale(lr)

        psnr_val = calculate_psnr(hr, bicubic)
        ssim_val = calculate_ssim(hr, bicubic)

        lr_np = tensor_to_numpy_image(lr, is_hr=False)
        bicubic_np = tensor_to_numpy_image(bicubic, is_hr=False)
        hr_np = tensor_to_numpy_image(hr, is_hr=False)

        # LR Column
        axes[row_idx, 0].imshow(lr_np)
        axes[row_idx, 0].set_title(f"Low-Resolution (64x64)\nCategory: {cat}", fontsize=9)
        axes[row_idx, 0].axis("off")

        # Bicubic Column
        axes[row_idx, 1].imshow(bicubic_np)
        axes[row_idx, 1].set_title(f"Bicubic 4x (256x256)\nPSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}", fontsize=9)
        axes[row_idx, 1].axis("off")

        # HR Column
        axes[row_idx, 2].imshow(hr_np)
        axes[row_idx, 2].set_title(f"Ground Truth HR (256x256)\nOriginal Satellite Image", fontsize=9)
        axes[row_idx, 2].axis("off")

    plt.suptitle("4x Satellite Super-Resolution: Bicubic Interpolation Baseline", fontsize=13, y=0.995)
    plt.tight_layout()

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved visual comparison grid to: {output_image_path}")


def main():
    test_ds = SatelliteSRDataset(split="test")
    summary = evaluate_bicubic(test_ds)
    create_visual_comparison_grid(test_ds)


if __name__ == "__main__":
    main()
