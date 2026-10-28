"""
Automated Test Suite for SRGAN Models (Generator, Discriminator) and Loss Functions.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
import torch.nn as nn

from src import config
from src.models.generator import Generator, ResidualBlock, PixelShuffleBlock
from src.models.discriminator import Discriminator, DiscriminatorBlock
from src.models.loss import VGGPerceptualLoss, GeneratorLoss, calculate_discriminator_loss


class TestGeneratorArchitecture:
    """Tests for Generator structure, shapes, and normalization contracts."""

    def test_generator_output_shape_and_range(self):
        batch_size = 4
        lr_size = 64
        hr_size = 256
        
        # Generator takes LR input in [0, 1] range
        netG = Generator(num_features=32, num_residual_blocks=4)  # Small size for fast test
        netG.eval()
        
        x = torch.rand(batch_size, 3, lr_size, lr_size, dtype=torch.float32)
        with torch.no_grad():
            out = netG(x)
            
        assert out.shape == (batch_size, 3, hr_size, hr_size), f"Expected {(batch_size, 3, hr_size, hr_size)}, got {out.shape}"
        assert out.dtype == torch.float32
        assert -1.0 <= out.min().item() and out.max().item() <= 1.0, (
            f"Output out of [-1, 1] range: min={out.min().item()}, max={out.max().item()}"
        )

    def test_generator_configurability(self):
        # Test custom channel and block count
        netG = Generator(in_channels=3, out_channels=3, num_features=16, num_residual_blocks=2, upscale_factor=4)
        x = torch.rand(2, 3, 64, 64)
        out = netG(x)
        assert out.shape == (2, 3, 256, 256)


class TestDiscriminatorArchitecture:
    """Tests for Discriminator structure, shapes, and logit outputs."""

    def test_discriminator_output_shape_and_logits(self):
        batch_size = 4
        hr_size = 256
        
        netD = Discriminator(in_channels=3, base_channels=32)
        netD.eval()
        
        # Discriminator takes images in [-1, 1] range
        x = torch.empty(batch_size, 3, hr_size, hr_size).uniform_(-1.0, 1.0)
        with torch.no_grad():
            logits = netD(x)
            probs = netD.predict_probability(x)
            
        assert logits.shape == (batch_size, 1), f"Expected logits shape {(batch_size, 1)}, got {logits.shape}"
        assert probs.shape == (batch_size, 1), f"Expected probs shape {(batch_size, 1)}, got {probs.shape}"
        assert torch.isfinite(logits).all(), "Discriminator logits contain non-finite values"
        assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities out of [0, 1]"


class TestLossFunctions:
    """Tests for perceptual loss, adversarial loss, parameter freezing, and gradients."""

    def test_vgg_loss_freezing_and_gradient_flow(self):
        vgg_loss = VGGPerceptualLoss(feature_layer=35, pretrained=False)
        
        # Verify all VGG parameters are strictly frozen
        for name, param in vgg_loss.named_parameters():
            assert not param.requires_grad, f"VGG parameter {name} has requires_grad=True!"

        # Input tensors in [-1, 1]
        sr = torch.randn(2, 3, 256, 256, requires_grad=True)
        hr = torch.randn(2, 3, 256, 256)
        
        loss = vgg_loss(sr, hr)
        assert torch.isfinite(loss), f"VGG loss is not finite: {loss.item()}"
        assert loss.item() >= 0.0, f"MSE perceptual loss should be non-negative: {loss.item()}"
        
        # Verify gradients flow back through the generated SR image
        loss.backward()
        assert sr.grad is not None, "Gradients did not flow back to generated image!"
        assert torch.isfinite(sr.grad).all(), "Gradients through SR contain non-finite values!"
        assert (sr.grad.abs().sum() > 0), "Gradients through SR are all zeros!"

    def test_generator_loss_terms_and_backward(self):
        batch_size = 2
        netG = Generator(num_features=32, num_residual_blocks=2)
        netD = Discriminator(base_channels=32)
        criterion_G = GeneratorLoss(content_weight=1.0, vgg_weight=1.0, adv_weight=1e-3, pretrained_vgg=False)

        lr = torch.rand(batch_size, 3, 64, 64)
        hr = torch.empty(batch_size, 3, 256, 256).uniform_(-1.0, 1.0)

        sr = netG(lr)
        disc_fake_logits = netD(sr)

        total_loss, content_loss, vgg_loss, adv_loss = criterion_G(sr, hr, disc_fake_logits)

        # Assert all terms are finite
        assert torch.isfinite(total_loss), "Total generator loss is not finite"
        assert torch.isfinite(content_loss), "Content loss is not finite"
        assert torch.isfinite(vgg_loss), "VGG loss is not finite"
        assert torch.isfinite(adv_loss), "Adversarial loss is not finite"

        # Verify backward pass populates Generator gradients
        total_loss.backward()
        for name, param in netG.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Generator parameter {name} has no gradient!"
                assert torch.isfinite(param.grad).all(), f"Generator parameter {name} has non-finite gradient!"

    def test_discriminator_loss_and_backward(self):
        batch_size = 2
        netD = Discriminator(base_channels=32)
        
        real_img = torch.empty(batch_size, 3, 256, 256).uniform_(-1.0, 1.0)
        fake_img = torch.empty(batch_size, 3, 256, 256).uniform_(-1.0, 1.0)
        
        real_logits = netD(real_img)
        fake_logits = netD(fake_img.detach())
        
        d_loss, loss_real, loss_fake = calculate_discriminator_loss(real_logits, fake_logits)
        
        assert torch.isfinite(d_loss), "Discriminator loss is not finite"
        assert torch.isfinite(loss_real), "D real loss is not finite"
        assert torch.isfinite(loss_fake), "D fake loss is not finite"
        
        d_loss.backward()
        for name, param in netD.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Discriminator parameter {name} has no gradient!"
                assert torch.isfinite(param.grad).all(), f"Discriminator parameter {name} has non-finite gradient!"


class TestCUDAPassAndMixedPrecision:
    """Tests forward and backward passes on CUDA hardware with AMP if available."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward_backward_with_amp(self):
        device = torch.device("cuda:0")
        batch_size = 4
        
        netG = Generator(num_features=32, num_residual_blocks=4).to(device)
        netD = Discriminator(base_channels=32).to(device)
        criterion_G = GeneratorLoss(pretrained_vgg=False).to(device)
        
        scaler = torch.amp.GradScaler("cuda")
        
        lr = torch.rand(batch_size, 3, 64, 64, device=device)
        hr = torch.empty(batch_size, 3, 256, 256, device=device).uniform_(-1.0, 1.0)
        
        # Test AMP forward pass
        with torch.amp.autocast("cuda", dtype=torch.float16):
            sr = netG(lr)
            disc_fake = netD(sr)
            total_loss, _, _, _ = criterion_G(sr, hr, disc_fake)
            
        assert sr.is_cuda, "SR output is not on CUDA"
        assert torch.isfinite(total_loss), "CUDA AMP loss is not finite"
        
        # Test Scaler backward pass
        scaler.scale(total_loss).backward()
        
        # Verify valid gradients
        has_grad = False
        for param in netG.parameters():
            if param.grad is not None:
                has_grad = True
                assert torch.isfinite(param.grad).all(), "CUDA gradient contains NaN/Inf"
        assert has_grad, "No gradients were computed on CUDA!"
