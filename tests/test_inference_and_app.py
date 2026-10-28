"""
Tests for inference utilities and Streamlit application helper functions.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
from PIL import Image
import torch

from src.inference import (
    validate_and_preprocess_image,
    pil_to_lr_tensor,
    tensor_to_pil,
    run_super_resolution,
    compute_image_metrics,
)
from src.models.generator import Generator


class TestInferencePipeline:
    """Test inference helpers, validation, and preprocessing."""

    def test_validate_and_preprocess_64x64(self):
        """Test with an exact 64x64 LR input."""
        img_64 = Image.new("RGB", (64, 64), color="blue")
        lr, hr, note = validate_and_preprocess_image(img_64)
        assert lr.size == (64, 64)
        assert hr is None
        assert "matches low-resolution format" in note

    def test_validate_and_preprocess_256x256(self):
        """Test with a 256x256 HR input."""
        img_256 = Image.new("RGB", (256, 256), color="red")
        lr, hr, note = validate_and_preprocess_image(img_256)
        assert lr.size == (64, 64)
        assert hr is not None
        assert hr.size == (256, 256)
        assert "high-resolution" in note

    def test_validate_and_preprocess_rgba_arbitrary_dimension(self):
        """Test mode conversion from RGBA to RGB and arbitrary dimensions."""
        img_rgba = Image.new("RGBA", (120, 180), color=(100, 150, 200, 255))
        lr, hr, note = validate_and_preprocess_image(img_rgba)
        assert lr.mode == "RGB"
        assert lr.size == (64, 64)
        assert hr is not None
        assert hr.mode == "RGB"
        assert hr.size == (256, 256)

    def test_pil_to_lr_tensor_and_tensor_to_pil(self):
        """Test tensor conversion and shape/range preservation."""
        img = Image.new("RGB", (64, 64), color=(128, 64, 32))
        tensor = pil_to_lr_tensor(img)
        assert tensor.shape == (1, 3, 64, 64)
        assert 0.0 <= tensor.min().item()
        assert tensor.max().item() <= 1.0

        recovered_pil = tensor_to_pil(tensor.squeeze(0), is_hr=False)
        assert recovered_pil.size == (64, 64)
        assert recovered_pil.mode == "RGB"

    def test_run_super_resolution_execution(self):
        """Test end-to-end forward super-resolution execution with generator models."""
        device = torch.device("cpu")
        model_pretrain = Generator(num_residual_blocks=2).to(device)
        model_srgan = Generator(num_residual_blocks=2).to(device)
        model_pretrain.eval()
        model_srgan.eval()

        lr_img = Image.new("RGB", (64, 64), color="green")
        outputs = run_super_resolution(lr_img, model_pretrain, model_srgan, device)

        assert "bicubic" in outputs
        assert "pretrain" in outputs
        assert "srgan" in outputs

        for k, v in outputs.items():
            assert isinstance(v, Image.Image)
            assert v.size == (256, 256)
            assert v.mode == "RGB"

    def test_compute_image_metrics(self):
        """Test metric calculation on identical and differing images."""
        img1 = Image.new("RGB", (256, 256), color=(100, 100, 100))
        img2 = Image.new("RGB", (256, 256), color=(100, 100, 100))

        identical_m = compute_image_metrics(img1, img2)
        assert identical_m["psnr"] == float("inf")
        assert identical_m["ssim"] == 1.0
        assert identical_m["mse"] == 0.0

        img3 = Image.new("RGB", (256, 256), color=(120, 120, 120))
        diff_m = compute_image_metrics(img1, img3)
        assert 0.0 < diff_m["psnr"] < float("inf")
        assert 0.0 <= diff_m["ssim"] <= 1.0
        assert diff_m["mse"] > 0.0
