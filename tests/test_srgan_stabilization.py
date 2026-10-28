"""
Tests for SRGAN Stabilization Configuration, Loss Formulations, and Optimizer Dynamics.
Verifies:
1. Generator optimizer LR is 1e-4, Discriminator optimizer LR is 2e-5.
2. Discriminator real targets are smoothed to 0.9, fake targets remain 0.0.
3. Generator adversarial targets strictly remain 1.0 (no accidental smoothing).
4. Discriminator and Generator backward passes and gradient health.
5. Mixed-precision (AMP) execution on CUDA if available.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src import config
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.models.loss import GeneratorLoss, calculate_discriminator_loss
from src.train_srgan import save_srgan_checkpoints, load_srgan_checkpoints


class TestSRGANStabilizationParameters:
    def test_configuration_learning_rates_and_smoothing(self):
        """Verify configuration constants match the approved stabilization plan."""
        assert config.GENERATOR_LR == 1e-4, f"Expected GENERATOR_LR 1e-4, got {config.GENERATOR_LR}"
        assert config.DISCRIMINATOR_LR == 2e-5, f"Expected DISCRIMINATOR_LR 2e-5, got {config.DISCRIMINATOR_LR}"
        assert config.DISCRIMINATOR_LABEL_SMOOTHING == 0.1, (
            f"Expected DISCRIMINATOR_LABEL_SMOOTHING 0.1, got {config.DISCRIMINATOR_LABEL_SMOOTHING}"
        )
        assert config.ADVERSARIAL_WEIGHT == 1e-3, f"Expected ADVERSARIAL_WEIGHT 1e-3, got {config.ADVERSARIAL_WEIGHT}"

    def test_optimizer_learning_rates(self):
        """Verify generator and discriminator optimizers instantiate with distinct target LRs."""
        netG = Generator(num_features=32, num_residual_blocks=2)
        netD = Discriminator(base_channels=32)

        optG = torch.optim.Adam(netG.parameters(), lr=config.GENERATOR_LR, betas=(0.9, 0.999))
        optD = torch.optim.Adam(netD.parameters(), lr=config.DISCRIMINATOR_LR, betas=(0.9, 0.999))

        for param_group in optG.param_groups:
            assert param_group["lr"] == 1e-4, f"optG LR should be 1e-4, got {param_group['lr']}"

        for param_group in optD.param_groups:
            assert param_group["lr"] == 2e-5, f"optD LR should be 2e-5, got {param_group['lr']}"

    def test_discriminator_label_smoothing_targets(self):
        """
        Verify calculate_discriminator_loss with label_smoothing=0.1
        evaluates against real target 0.9 and fake target 0.0.
        """
        real_logits = torch.tensor([[0.0], [2.0], [-1.0]], requires_grad=True)
        fake_logits = torch.tensor([[-2.0], [0.0], [1.0]], requires_grad=True)

        d_loss, loss_real, loss_fake = calculate_discriminator_loss(
            real_logits, fake_logits, label_smoothing=0.1
        )

        bce = nn.BCEWithLogitsLoss()
        expected_real = bce(real_logits, torch.full_like(real_logits, 0.9))
        expected_fake = bce(fake_logits, torch.zeros_like(fake_logits))
        expected_total = 0.5 * (expected_real + expected_fake)

        assert torch.allclose(loss_real, expected_real), "loss_real does not match target 0.9"
        assert torch.allclose(loss_fake, expected_fake), "loss_fake does not match target 0.0"
        assert torch.allclose(d_loss, expected_total), "d_loss does not match 0.5*(real+fake)"

    def test_generator_adversarial_targets_remain_one(self):
        """
        Verify GeneratorLoss adversarial loss targets remain hard 1.0
        and are NOT smoothed.
        """
        criterion_G = GeneratorLoss(content_weight=1.0, vgg_weight=1.0, adv_weight=1e-3, pretrained_vgg=False)
        disc_fake_logits = torch.tensor([[-3.0], [0.0], [2.0]], requires_grad=True)
        sr = torch.randn(3, 3, 256, 256, requires_grad=True)
        hr = torch.randn(3, 3, 256, 256)

        total_loss, content_loss, vgg_loss, adv_loss = criterion_G(sr, hr, disc_fake_logits)

        bce = nn.BCEWithLogitsLoss()
        expected_adv = bce(disc_fake_logits, torch.ones_like(disc_fake_logits))

        assert torch.allclose(adv_loss, expected_adv), "Generator adversarial loss did not target hard 1.0!"

    def test_vgg_weight_scaling(self):
        """
        Verify GeneratorLoss properly scales VGG loss with configurable vgg_weight (e.g. 0.1).
        total_loss = content_weight * content + vgg_weight * vgg + adv_weight * adv
        """
        criterion_G = GeneratorLoss(content_weight=1.0, vgg_weight=0.1, adv_weight=1e-3, pretrained_vgg=False)
        disc_fake_logits = torch.tensor([[-1.0], [0.5]], requires_grad=True)
        sr = torch.randn(2, 3, 256, 256, requires_grad=True)
        hr = torch.randn(2, 3, 256, 256)

        total_loss, content_loss, vgg_loss, adv_loss = criterion_G(sr, hr, disc_fake_logits)

        expected_total = 1.0 * content_loss + 0.1 * vgg_loss + 1e-3 * adv_loss
        assert torch.allclose(total_loss, expected_total, atol=1e-6), (
            f"Expected {expected_total.item()}, got {total_loss.item()}"
        )

    def test_generator_and_discriminator_backward_passes(self):
        """Verify backward passes compute finite, non-zero gradients under stabilization settings."""
        netG = Generator(num_features=32, num_residual_blocks=2)
        netD = Discriminator(base_channels=32)

        lr = torch.rand(2, 3, 64, 64)
        hr = torch.empty(2, 3, 256, 256).uniform_(-1.0, 1.0)

        # 1. Step D with label smoothing
        sr_fake = netG(lr).detach()
        real_logits = netD(hr)
        fake_logits = netD(sr_fake)
        d_loss, loss_real, loss_fake = calculate_discriminator_loss(
            real_logits, fake_logits, label_smoothing=config.DISCRIMINATOR_LABEL_SMOOTHING
        )
        assert torch.isfinite(d_loss)
        d_loss.backward()

        for name, p in netD.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Discriminator param {name} missing grad"
                assert torch.isfinite(p.grad).all(), f"Discriminator param {name} has non-finite grad"

        # 2. Step G with adversarial target 1.0
        criterion_G = GeneratorLoss(content_weight=1.0, vgg_weight=1.0, adv_weight=1e-3, pretrained_vgg=False)
        sr_gen = netG(lr)
        disc_fake = netD(sr_gen)
        g_loss, c_loss, v_loss, a_loss = criterion_G(sr_gen, hr, disc_fake)
        assert torch.isfinite(g_loss)
        g_loss.backward()

        for name, p in netG.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Generator param {name} missing grad"
                assert torch.isfinite(p.grad).all(), f"Generator param {name} has non-finite grad"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision_stabilized_step(self):
        """Verify CUDA mixed precision forward and backward passes execute cleanly."""
        device = torch.device("cuda")
        netG = Generator(num_features=32, num_residual_blocks=2).to(device)
        netD = Discriminator(base_channels=32).to(device)

        optG = torch.optim.Adam(netG.parameters(), lr=config.GENERATOR_LR)
        optD = torch.optim.Adam(netD.parameters(), lr=config.DISCRIMINATOR_LR)
        scalerG = torch.amp.GradScaler("cuda")
        scalerD = torch.amp.GradScaler("cuda")

        lr = torch.rand(2, 3, 64, 64, device=device)
        hr = torch.empty(2, 3, 256, 256, device=device).uniform_(-1.0, 1.0)

        # D AMP step
        optD.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            sr_fake = netG(lr).detach()
            real_logits = netD(hr)
            fake_logits = netD(sr_fake)
            d_loss, _, _ = calculate_discriminator_loss(
                real_logits, fake_logits, label_smoothing=0.1
            )
        assert torch.isfinite(d_loss)
        scalerD.scale(d_loss).backward()
        scalerD.step(optD)
        scalerD.update()

        # G AMP step
        criterion_G = GeneratorLoss(pretrained_vgg=False).to(device)
        optG.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            sr_gen = netG(lr)
            disc_fake = netD(sr_gen)
            g_loss, _, _, _ = criterion_G(sr_gen, hr, disc_fake)
        assert torch.isfinite(g_loss)
        scalerG.scale(g_loss).backward()
        scalerG.step(optG)
        scalerG.update()

    def test_two_generator_to_one_discriminator_schedule(self):
        """
        Verify the 2:1 update cadence:
        1. Discriminator only steps when (batch_idx % d_interval == 0).
        2. Discriminator parameters accumulate NO gradients during Generator-only updates.
        3. Generator updates on every batch.
        4. Gradient flows through D to G even when D.requires_grad is temporarily False.
        """
        netG = Generator(num_features=32, num_residual_blocks=2)
        netD = Discriminator(base_channels=32)

        optG = torch.optim.Adam(netG.parameters(), lr=1e-4)
        optD = torch.optim.Adam(netD.parameters(), lr=2e-5)
        criterion_G = GeneratorLoss(content_weight=1.0, vgg_weight=0.1, adv_weight=1e-3, pretrained_vgg=False)

        d_interval = 2
        d_step_count = 0
        g_step_count = 0

        for batch_idx in range(4):
            lr = torch.rand(2, 3, 64, 64)
            hr = torch.empty(2, 3, 256, 256).uniform_(-1.0, 1.0)
            update_d = (batch_idx % d_interval == 0)

            if update_d:
                optD.zero_grad()
                sr_fake = netG(lr).detach()
                real_logits = netD(hr)
                fake_logits = netD(sr_fake)
                d_loss, _, _ = calculate_discriminator_loss(real_logits, fake_logits, label_smoothing=0.1)
                d_loss.backward()
                optD.step()
                d_step_count += 1
                # After D step, clear D grads for clean tracking
                optD.zero_grad()

            # G Update: Every batch
            optG.zero_grad()
            for p in netD.parameters():
                p.requires_grad = False

            sr_gen = netG(lr)
            disc_fake = netD(sr_gen)
            g_loss, _, _, _ = criterion_G(sr_gen, hr, disc_fake)
            g_loss.backward()
            optG.step()
            g_step_count += 1

            for p in netD.parameters():
                p.requires_grad = True

            # Verify that Discriminator parameters have NO gradients accumulated from G's backward pass
            for name, param in netD.named_parameters():
                assert param.grad is None or (param.grad.abs().sum() == 0), (
                    f"Discriminator parameter {name} accumulated gradient during batch {batch_idx}!"
                )

        assert g_step_count == 4, f"Expected 4 G steps, got {g_step_count}"
        assert d_step_count == 2, f"Expected 2 D steps, got {d_step_count}"
        assert g_step_count / d_step_count == 2.0, "Expected exact 2:1 update ratio!"

    def test_srgan_checkpoint_resume_integrity(self, tmp_path):
        """
        Verify that save_srgan_checkpoints and load_srgan_checkpoints properly preserve
        and restore model state dicts, optimizer state dicts, scaler state dicts,
        epoch index, and validation PSNR.
        """
        netG = Generator(num_features=32, num_residual_blocks=2)
        netD = Discriminator(base_channels=32)
        optG = torch.optim.Adam(netG.parameters(), lr=1e-4)
        optD = torch.optim.Adam(netD.parameters(), lr=2e-5)

        # Run a dummy step to populate optimizer state
        dummy_x = torch.randn(2, 3, 64, 64)
        dummy_y = netG(dummy_x)
        dummy_y.sum().backward()
        optG.step()

        dummy_d = netD(dummy_y.detach())
        dummy_d.sum().backward()
        optD.step()

        # Save checkpoint
        save_srgan_checkpoints(
            epoch=1,
            model_G=netG,
            model_D=netD,
            optimizer_G=optG,
            optimizer_D=optD,
            scaler_G=None,
            scaler_D=None,
            val_psnr=24.92,
            val_ssim=0.6823,
            val_loss=0.01868,
            is_best=True,
            checkpoint_dir=tmp_path,
            prefix="test_srgan",
        )

        resume_g = tmp_path / "test_srgan_generator_best.pth"
        resume_d = tmp_path / "test_srgan_discriminator_best.pth"
        assert resume_g.exists()
        assert resume_d.exists()

        # Load into fresh models & optimizers
        fresh_G = Generator(num_features=32, num_residual_blocks=2)
        fresh_D = Discriminator(base_channels=32)
        fresh_optG = torch.optim.Adam(fresh_G.parameters(), lr=1e-4)
        fresh_optD = torch.optim.Adam(fresh_D.parameters(), lr=2e-5)

        start_epoch, loaded_psnr = load_srgan_checkpoints(
            resume_g, resume_d, fresh_G, fresh_D, fresh_optG, fresh_optD
        )

        assert start_epoch == 2, f"Expected resumed start_epoch 2 (from epoch index 1), got {start_epoch}"
        assert abs(loaded_psnr - 24.92) < 1e-4, f"Expected resumed PSNR 24.92, got {loaded_psnr}"
        assert len(fresh_optG.state) > 0, "fresh_optG state was not restored!"
        assert len(fresh_optD.state) > 0, "fresh_optD state was not restored!"
