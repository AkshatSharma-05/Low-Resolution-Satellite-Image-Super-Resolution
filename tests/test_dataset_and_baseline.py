"""
Automated Test Suite for Dataset Pipeline, Splits, and Baseline Metrics.
"""

from pathlib import Path
import sys
import json

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import torch
from PIL import Image

from src import config
from src.dataset import (
    discover_images,
    create_stratified_splits,
    degrade_to_lr,
    denormalize_hr,
    denormalize_lr,
    tensor_to_numpy_image,
    SatelliteSRDataset,
)
from src.metrics import calculate_psnr, calculate_ssim
from src.baseline import bicubic_upscale


class TestDatasetDiscoveryAndSplits:
    """Tests for dataset discovery, class distribution, and split disjointness."""
    
    def test_dataset_discovery(self):
        categories = discover_images(config.RAW_IMAGES_DIR)
        assert len(categories) == 21, f"Expected 21 categories, got {len(categories)}"
        
        total_images = sum(len(v) for v in categories.values())
        assert total_images == 2100, f"Expected 2100 total images, got {total_images}"
        
        for cat_name, file_list in categories.items():
            assert len(file_list) == 100, f"Category '{cat_name}' has {len(file_list)} images, expected 100."
            for p in file_list:
                assert p.exists(), f"Image file not found: {p}"
                assert p.suffix.lower() in {".tif", ".tiff"}, f"Unexpected file extension: {p.suffix}"

    def test_split_disjointness_and_completeness(self):
        splits = create_stratified_splits(splits_dir=config.SPLITS_DIR)
        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])

        # Check total counts
        assert len(train_set) == 1680, f"Expected 1680 train files, got {len(train_set)}"
        assert len(val_set) == 210, f"Expected 210 val files, got {len(val_set)}"
        assert len(test_set) == 210, f"Expected 210 test files, got {len(test_set)}"

        # Verify pairwise disjointness
        train_val_overlap = train_set.intersection(val_set)
        train_test_overlap = train_set.intersection(test_set)
        val_test_overlap = val_set.intersection(test_set)

        assert len(train_val_overlap) == 0, f"Overlap between train and val: {train_val_overlap}"
        assert len(train_test_overlap) == 0, f"Overlap between train and test: {train_test_overlap}"
        assert len(val_test_overlap) == 0, f"Overlap between val and test: {val_test_overlap}"

        # Verify union is complete
        total_union = train_set.union(val_set).union(test_set)
        assert len(total_union) == 2100, f"Union count {len(total_union)} != 2100"

    def test_class_stratification(self):
        splits = create_stratified_splits(splits_dir=config.SPLITS_DIR)
        categories = discover_images(config.RAW_IMAGES_DIR)

        for cat in categories.keys():
            train_cat_count = sum(1 for p in splits["train"] if p.startswith(cat + "/"))
            val_cat_count = sum(1 for p in splits["val"] if p.startswith(cat + "/"))
            test_cat_count = sum(1 for p in splits["test"] if p.startswith(cat + "/"))

            assert train_cat_count == 80, f"Category '{cat}' train count is {train_cat_count}, expected 80"
            assert val_cat_count == 10, f"Category '{cat}' val count is {val_cat_count}, expected 10"
            assert test_cat_count == 10, f"Category '{cat}' test count is {test_cat_count}, expected 10"


class TestImageDegradationAndTensors:
    """Tests for image degradation, tensor dimensions, and value ranges."""

    def test_image_pairing_and_pil_degradation(self):
        # Create a sample PIL Image of size (256, 256)
        img = Image.new("RGB", (256, 256), color=(128, 64, 32))
        lr_img = degrade_to_lr(img, lr_size=64)

        assert lr_img.size == (64, 64), f"Expected LR size (64, 64), got {lr_img.size}"
        assert lr_img.mode == "RGB", f"Expected RGB mode, got {lr_img.mode}"

    def test_pytorch_dataset_tensors(self):
        ds = SatelliteSRDataset(split="val", augment=False)
        assert len(ds) == 210

        lr_tensor, hr_tensor, info = ds[0]

        # Shape assertions
        assert lr_tensor.shape == (3, 64, 64), f"LR shape expected (3, 64, 64), got {lr_tensor.shape}"
        assert hr_tensor.shape == (3, 256, 256), f"HR shape expected (3, 256, 256), got {hr_tensor.shape}"

        # Data type assertions
        assert lr_tensor.dtype == torch.float32, f"LR dtype expected float32, got {lr_tensor.dtype}"
        assert hr_tensor.dtype == torch.float32, f"HR dtype expected float32, got {hr_tensor.dtype}"

        # Value range assertions
        assert 0.0 <= lr_tensor.min() and lr_tensor.max() <= 1.0, (
            f"LR out of [0, 1] range: min={lr_tensor.min()}, max={lr_tensor.max()}"
        )
        assert -1.0 <= hr_tensor.min() and hr_tensor.max() <= 1.0, (
            f"HR out of [-1, 1] range: min={hr_tensor.min()}, max={hr_tensor.max()}"
        )

        # Metadata dictionary
        assert "rel_path" in info
        assert "category" in info

    def test_normalization_and_invertibility(self):
        # Create tensor in [-1.0, 1.0]
        hr_t = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float32)
        denorm = denormalize_hr(hr_t)
        expected = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
        assert torch.allclose(denorm, expected, atol=1e-5), f"Denorm mismatch: {denorm} vs {expected}"

        # Convert 3-channel tensor to uint8 numpy
        fake_hr = torch.ones(3, 256, 256, dtype=torch.float32)  # In [-1, 1], 1.0 corresponds to 255
        np_img = tensor_to_numpy_image(fake_hr, is_hr=True)
        assert np_img.shape == (256, 256, 3)
        assert np_img.dtype == np.uint8
        assert np.all(np_img == 255), f"Expected all 255, got min={np_img.min()}"


class TestMetricsAndBaseline:
    """Tests for metric calculations and bicubic baseline."""

    def test_psnr_and_ssim_identical_images(self):
        img = np.random.uniform(0.0, 1.0, (256, 256, 3)).astype(np.float32)
        psnr_val = calculate_psnr(img, img)
        ssim_val = calculate_ssim(img, img)

        assert psnr_val == float("inf"), f"PSNR of identical images should be inf, got {psnr_val}"
        assert abs(ssim_val - 1.0) < 1e-4, f"SSIM of identical images should be 1.0, got {ssim_val}"

    def test_psnr_and_ssim_degraded_image(self):
        img1 = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        img2 = img1 + np.random.normal(0, 0.05, img1.shape).astype(np.float32)
        img2 = np.clip(img2, 0.0, 1.0)

        psnr_val = calculate_psnr(img1, img2)
        ssim_val = calculate_ssim(img1, img2)

        assert 15.0 < psnr_val < 45.0, f"Expected realistic PSNR, got {psnr_val}"
        assert 0.0 < ssim_val < 1.0, f"Expected SSIM in (0, 1), got {ssim_val}"

    def test_bicubic_upscale(self):
        lr_tensor = torch.rand(3, 64, 64, dtype=torch.float32)
        upscaled = bicubic_upscale(lr_tensor, target_size=256)

        assert upscaled.shape == (3, 256, 256), f"Expected (3, 256, 256), got {upscaled.shape}"
        assert 0.0 <= upscaled.min() and upscaled.max() <= 1.0, "Upscaled tensor out of [0, 1]"
