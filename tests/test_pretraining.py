"""
Automated Test Suite for Generator Pretraining:
Tests single training step, checkpoint save/load, and validation metric computation.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src import config
from src.dataset import SatelliteSRDataset
from src.models.generator import Generator
from src.train_pretrain import validate_generator, save_checkpoint, load_checkpoint


class TestPretrainingPipeline:
    """Unit and integration tests for SRResNet pretraining routines."""

    def test_one_training_step(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        netG = Generator(num_features=32, num_residual_blocks=2).to(device)
        optimizer = torch.optim.Adam(netG.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        lr = torch.rand(2, 3, 64, 64, device=device)
        hr = torch.empty(2, 3, 256, 256, device=device).uniform_(-1.0, 1.0)

        # Store initial weight of a conv layer to verify update
        initial_param = netG.initial_conv[0].weight.clone()

        netG.train()
        optimizer.zero_grad()
        sr = netG(lr)
        loss = criterion(sr, hr)
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss), "Training step loss is not finite"
        updated_param = netG.initial_conv[0].weight
        assert not torch.equal(initial_param, updated_param), "Weights did not change after optimizer step!"

    def test_checkpoint_save_and_load(self, tmp_path):
        device = torch.device("cpu")
        model1 = Generator(num_features=32, num_residual_blocks=2)
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-4)

        # Create dummy state
        state = {
            "epoch": 3,
            "model_state_dict": model1.state_dict(),
            "optimizer_state_dict": optimizer1.state_dict(),
            "val_psnr": 26.5,
            "val_ssim": 0.72,
        }
        
        save_checkpoint(state, is_best=True, checkpoint_dir=tmp_path)
        
        latest_file = tmp_path / "generator_pretrain_latest.pth"
        best_file = tmp_path / "generator_pretrain_best.pth"
        assert latest_file.exists(), "Latest checkpoint was not saved!"
        assert best_file.exists(), "Best checkpoint was not saved!"

        # Load into fresh model
        model2 = Generator(num_features=32, num_residual_blocks=2)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
        loaded_state = load_checkpoint(best_file, model2, optimizer2, device=device)

        assert loaded_state["epoch"] == 3
        assert loaded_state["val_psnr"] == 26.5

        # Verify inference outputs are identical
        test_in = torch.rand(1, 3, 64, 64)
        model1.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model1(test_in)
            out2 = model2(test_in)
            
        assert torch.allclose(out1, out2, atol=1e-6), "Loaded model produces different output from original!"

    def test_validation_metrics_finite(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        netG = Generator(num_features=32, num_residual_blocks=2).to(device)
        criterion = nn.MSELoss()

        val_ds = SatelliteSRDataset(split="val", augment=False)
        # Use a small subset (4 samples = 1 batch) for fast test execution
        subset_loader = DataLoader(Subset(val_ds, [0, 1, 2, 3]), batch_size=4)

        metrics = validate_generator(netG, subset_loader, criterion, device, use_amp=False)

        assert torch.isfinite(torch.tensor(metrics["val_loss"])), "Val loss is not finite"
        assert torch.isfinite(torch.tensor(metrics["gen_psnr"])), "Generator PSNR is not finite"
        assert torch.isfinite(torch.tensor(metrics["gen_ssim"])), "Generator SSIM is not finite"
        assert torch.isfinite(torch.tensor(metrics["bicubic_psnr"])), "Bicubic PSNR is not finite"
        assert torch.isfinite(torch.tensor(metrics["bicubic_ssim"])), "Bicubic SSIM is not finite"
        assert metrics["gen_psnr"] > 0, "PSNR should be positive"
        assert 0.0 <= metrics["gen_ssim"] <= 1.0, "SSIM should be in [0, 1]"
