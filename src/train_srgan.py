"""
SRGAN Adversarial Training Pipeline.
Jointly trains the SRResNet Generator and Discriminator with:
- Content Loss (MSE)
- VGG-19 Perceptual Loss (Deep Feature Representations)
- Adversarial Loss (BCEWithLogitsLoss)

Includes live unbuffered tqdm progress bars, separate loss logging (Content, VGG, Adv),
validation metric tracking (PSNR, SSIM), checkpoint saving, and training resumption.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout is unbuffered for live streaming in terminal with UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    except Exception:
        sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import time
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.dataset import SatelliteSRDataset, denormalize_hr
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.models.loss import GeneratorLoss, calculate_discriminator_loss
from src.baseline import bicubic_upscale
from src.metrics import calculate_psnr, calculate_ssim


def validate_srgan(
    model_G: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Evaluate Generator during adversarial training on the validation set.
    """
    model_G.eval()
    val_losses = []
    gen_psnrs = []
    gen_ssims = []
    bicubic_psnrs = []
    bicubic_ssims = []

    loader = tqdm(
        val_loader,
        desc="Validation",
        file=sys.stdout,
        leave=False,
        dynamic_ncols=True,
    ) if show_progress else val_loader

    with torch.no_grad():
        for lr, hr, _ in loader:
            lr = lr.to(device)
            hr = hr.to(device)

            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    sr = model_G(lr)
                    loss = nn.functional.mse_loss(sr, hr)
            else:
                sr = model_G(lr)
                loss = nn.functional.mse_loss(sr, hr)

            val_losses.append(loss.item())

            sr_01 = denormalize_hr(sr)
            hr_01 = denormalize_hr(hr)
            bicubic_01 = bicubic_upscale(lr, target_size=config.HR_SIZE)

            for b in range(lr.size(0)):
                psnr_sr = calculate_psnr(hr_01[b], sr_01[b])
                ssim_sr = calculate_ssim(hr_01[b], sr_01[b])
                gen_psnrs.append(psnr_sr)
                gen_ssims.append(ssim_sr)

                psnr_bi = calculate_psnr(hr_01[b], bicubic_01[b])
                ssim_bi = calculate_ssim(hr_01[b], bicubic_01[b])
                bicubic_psnrs.append(psnr_bi)
                bicubic_ssims.append(ssim_bi)

    return {
        "val_loss": float(np.mean(val_losses)),
        "gen_psnr": float(np.mean(gen_psnrs)),
        "gen_ssim": float(np.mean(gen_ssims)),
        "bicubic_psnr": float(np.mean(bicubic_psnrs)),
        "bicubic_ssim": float(np.mean(bicubic_ssims)),
    }


def save_srgan_checkpoints(
    epoch: int,
    model_G: nn.Module,
    model_D: nn.Module,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    scaler_G: Optional[torch.amp.GradScaler],
    scaler_D: Optional[torch.amp.GradScaler],
    val_psnr: float,
    val_ssim: float,
    val_loss: float,
    is_best: bool,
    checkpoint_dir: Path = config.CHECKPOINTS_DIR,
    prefix: str = "srgan",
) -> Tuple[Path, Optional[Path]]:
    """Save both Generator and Discriminator checkpoints."""
    if not isinstance(checkpoint_dir, Path):
        checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (PROJECT_ROOT / checkpoint_dir).resolve()
    else:
        checkpoint_dir = checkpoint_dir.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state_G = {
        "epoch": epoch,
        "model_state_dict": model_G.state_dict(),
        "optimizer_state_dict": optimizer_G.state_dict(),
        "scaler_state_dict": scaler_G.state_dict() if scaler_G else None,
        "val_psnr": val_psnr,
        "val_ssim": val_ssim,
        "val_loss": val_loss,
    }
    state_D = {
        "epoch": epoch,
        "model_state_dict": model_D.state_dict(),
        "optimizer_state_dict": optimizer_D.state_dict(),
        "scaler_state_dict": scaler_D.state_dict() if scaler_D else None,
    }

    latest_G = checkpoint_dir / f"{prefix}_generator_latest.pth"
    latest_D = checkpoint_dir / f"{prefix}_discriminator_latest.pth"
    torch.save(state_G, latest_G)
    torch.save(state_D, latest_D)

    best_G = None
    if is_best:
        best_G = checkpoint_dir / f"{prefix}_generator_best.pth"
        best_D = checkpoint_dir / f"{prefix}_discriminator_best.pth"
        torch.save(state_G, best_G)
        torch.save(state_D, best_D)

    return latest_G, best_G


def load_srgan_checkpoints(
    resume_path_G: Path,
    resume_path_D: Optional[Path],
    model_G: nn.Module,
    model_D: nn.Module,
    optimizer_G: Optional[torch.optim.Optimizer] = None,
    optimizer_D: Optional[torch.optim.Optimizer] = None,
    scaler_G: Optional[torch.amp.GradScaler] = None,
    scaler_D: Optional[torch.amp.GradScaler] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, float]:
    """Load SRGAN checkpoints to resume adversarial training."""
    checkpoint_G = torch.load(resume_path_G, map_location=device)
    model_G.load_state_dict(checkpoint_G["model_state_dict"])
    if optimizer_G and "optimizer_state_dict" in checkpoint_G and checkpoint_G["optimizer_state_dict"] is not None:
        optimizer_G.load_state_dict(checkpoint_G["optimizer_state_dict"])
    if scaler_G and "scaler_state_dict" in checkpoint_G and checkpoint_G["scaler_state_dict"] is not None:
        scaler_G.load_state_dict(checkpoint_G["scaler_state_dict"])

    start_epoch = checkpoint_G.get("epoch", -1) + 1
    best_val_psnr = float(checkpoint_G.get("val_psnr", -float("inf")))

    if resume_path_D and resume_path_D.exists():
        checkpoint_D = torch.load(resume_path_D, map_location=device)
        model_D.load_state_dict(checkpoint_D["model_state_dict"])
        if optimizer_D and "optimizer_state_dict" in checkpoint_D and checkpoint_D["optimizer_state_dict"] is not None:
            optimizer_D.load_state_dict(checkpoint_D["optimizer_state_dict"])
        if scaler_D and "scaler_state_dict" in checkpoint_D and checkpoint_D["scaler_state_dict"] is not None:
            scaler_D.load_state_dict(checkpoint_D["scaler_state_dict"])
    else:
        raise FileNotFoundError(f"Matching Discriminator checkpoint not found at: {resume_path_D}")

    return start_epoch, best_val_psnr


def run_srgan_training(
    epochs: int = 25,
    batch_size: int = config.BATCH_SIZE,
    lr_g: float = config.GENERATOR_LR,
    lr_d: float = config.DISCRIMINATOR_LR,
    label_smoothing: float = config.DISCRIMINATOR_LABEL_SMOOTHING,
    d_interval: int = config.DISCRIMINATOR_UPDATE_INTERVAL,
    content_weight: float = 1.0,
    vgg_weight: float = config.CONTENT_VGG_WEIGHT,
    adv_weight: float = config.ADVERSARIAL_WEIGHT,
    pretrained_generator_path: Optional[str] = "checkpoints/generator_pretrain_best.pth",
    pretrained_vgg: bool = False,
    resume_path: Optional[str] = None,
    use_amp: bool = config.USE_AMP,
    checkpoint_dir: Path = config.CHECKPOINTS_DIR,
    checkpoint_prefix: str = "srgan",
    history_file: Optional[str] = None,
    smoke_test: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Main routine for SRGAN adversarial training with live progress, separate loss breakdown,
    configurable discriminator update cadence, and clean epoch summaries.
    """
    if device is None:
        device = config.DEVICE

    if not isinstance(checkpoint_dir, Path):
        checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (PROJECT_ROOT / checkpoint_dir).resolve()
    else:
        checkpoint_dir = checkpoint_dir.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print("SRGAN Adversarial Training", flush=True)
    print(
        f"Device: {device_name} | Batch Size: {batch_size} | Epochs: {epochs} | "
        f"LR_G: {lr_g} | LR_D: {lr_d} | D Interval: 1 D step per {d_interval} G steps | "
        f"D Label Smoothing: {label_smoothing} | Adv Weight: {adv_weight} | "
        f"Mixed Precision: {use_amp and device.type == 'cuda'}\n",
        flush=True,
    )

    # 1. Initialize Datasets & DataLoaders
    train_dataset = SatelliteSRDataset(split="train", augment=True)
    val_dataset = SatelliteSRDataset(split="val", augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    # 2. Initialize Models
    model_G = Generator(
        in_channels=3,
        out_channels=3,
        num_features=64,
        num_residual_blocks=16,
        upscale_factor=4,
    ).to(device)

    model_D = Discriminator(
        in_channels=3,
        base_channels=64,
    ).to(device)

    # Load Pretrained Generator Weights
    if pretrained_generator_path and not resume_path:
        ckpt_path = PROJECT_ROOT / pretrained_generator_path
        if ckpt_path.exists():
            print(f"Loading pretrained generator from: {ckpt_path.relative_to(PROJECT_ROOT).as_posix()}", flush=True)
            checkpoint = torch.load(ckpt_path, map_location=device)
            model_G.load_state_dict(checkpoint["model_state_dict"], strict=True)
            pretrain_val_psnr = checkpoint.get("val_psnr", 0.0)
            print(f"Pretrained generator successfully loaded (Pretrain Val PSNR: {pretrain_val_psnr:.2f} dB)\n", flush=True)
        else:
            print(f"[NOTE] Pretrained checkpoint {pretrained_generator_path} not found. Starting Generator from scratch.\n", flush=True)

    # 3. Optimizers, Loss Functions, Scalers
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=lr_g, betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=lr_d, betas=(0.9, 0.999))

    criterion_G = GeneratorLoss(
        content_weight=content_weight,
        vgg_weight=vgg_weight,
        adv_weight=adv_weight,
        pretrained_vgg=pretrained_vgg,
    ).to(device)

    scaler_G = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None
    scaler_D = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    start_epoch = 0
    best_val_psnr = -float("inf")
    prior_history = []

    # Resume from existing SRGAN checkpoint if provided
    if resume_path:
        resume_g = Path(resume_path)
        resume_d = resume_g.parent / resume_g.name.replace("generator", "discriminator")
        if not resume_g.exists():
            raise FileNotFoundError(f"Resume Generator checkpoint not found: {resume_g}")
        if not resume_d.exists():
            raise FileNotFoundError(f"Resume Discriminator checkpoint not found: {resume_d}")

        try:
            rel_g = resume_g.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            rel_g = str(resume_g)
        try:
            rel_d = resume_d.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            rel_d = str(resume_d)

        print(f"Resuming SRGAN training from:\n  Generator:     {rel_g}\n  Discriminator: {rel_d}", flush=True)

        start_epoch, loaded_best_psnr = load_srgan_checkpoints(
            resume_g, resume_d, model_G, model_D, optimizer_G, optimizer_D, scaler_G, scaler_D, device
        )
        if loaded_best_psnr > -float("inf"):
            best_val_psnr = loaded_best_psnr

        # Load prior history if available
        hist_candidate = resume_g.parent / f"{resume_g.name.split('_generator')[0]}_history.json"
        if hist_candidate.exists():
            try:
                with open(hist_candidate) as f:
                    full_hist = json.load(f)
                prior_history = [r for r in full_hist if r.get("epoch", 0) <= start_epoch]
                print(f"Loaded {len(prior_history)} prior epoch records from {hist_candidate.name}", flush=True)
            except Exception as e:
                print(f"[NOTE] Could not load prior history: {e}", flush=True)

        print(f"Resumed from epoch {start_epoch + 1} (Best Val PSNR so far: {best_val_psnr:.2f} dB)\n", flush=True)

    # Handle Smoke Test
    if smoke_test:
        print("Executing one-batch SRGAN adversarial smoke test...", flush=True)
        model_G.train()
        model_D.train()
        lr_sample, hr_sample, _ = next(iter(train_loader))
        lr_sample = lr_sample.to(device)
        hr_sample = hr_sample.to(device)

        # Step D
        optimizer_D.zero_grad()
        if scaler_D:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                sr_fake = model_G(lr_sample).detach()
                real_logits = model_D(hr_sample)
                fake_logits = model_D(sr_fake)
                d_loss, d_loss_real, d_loss_fake = calculate_discriminator_loss(
                    real_logits, fake_logits, label_smoothing=label_smoothing
                )
            if not torch.isfinite(d_loss):
                raise FloatingPointError(f"Smoke test Discriminator loss is non-finite: {d_loss.item()}")
            scaler_D.scale(d_loss).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()
        else:
            sr_fake = model_G(lr_sample).detach()
            real_logits = model_D(hr_sample)
            fake_logits = model_D(sr_fake)
            d_loss, d_loss_real, d_loss_fake = calculate_discriminator_loss(
                real_logits, fake_logits, label_smoothing=label_smoothing
            )
            if not torch.isfinite(d_loss):
                raise FloatingPointError(f"Smoke test Discriminator loss is non-finite: {d_loss.item()}")
            d_loss.backward()
            optimizer_D.step()

        # Step G
        optimizer_G.zero_grad()
        for p in model_D.parameters():
            p.requires_grad = False

        if scaler_G:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                sr_gen = model_G(lr_sample)
                disc_fake_logits = model_D(sr_gen)
                g_loss, c_loss, v_loss, a_loss = criterion_G(sr_gen, hr_sample, disc_fake_logits)
            if not torch.isfinite(g_loss):
                raise FloatingPointError(f"Smoke test Generator loss is non-finite: {g_loss.item()}")
            scaler_G.scale(g_loss).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()
        else:
            sr_gen = model_G(lr_sample)
            disc_fake_logits = model_D(sr_gen)
            g_loss, c_loss, v_loss, a_loss = criterion_G(sr_gen, hr_sample, disc_fake_logits)
            if not torch.isfinite(g_loss):
                raise FloatingPointError(f"Smoke test Generator loss is non-finite: {g_loss.item()}")
            g_loss.backward()
            optimizer_G.step()

        for p in model_D.parameters():
            p.requires_grad = True

        print(
            f"[SMOKE TEST SUCCESS] Loss G: {g_loss.item():.6f} "
            f"(Content: {c_loss.item():.6f} | VGG: {v_loss.item():.6f} | Adv: {a_loss.item():.6f} [Weighted: {a_loss.item()*adv_weight:.6f}]) | "
            f"Loss D: {d_loss.item():.6f} (Real: {d_loss_real.item():.6f} | Fake: {d_loss_fake.item():.6f})\n",
            flush=True,
        )
        return {
            "smoke_test_loss_G": g_loss.item(),
            "smoke_test_loss_D": d_loss.item(),
            "smoke_test_loss_D_real": d_loss_real.item(),
            "smoke_test_loss_D_fake": d_loss_fake.item(),
        }

    # 4. Training Loop
    total_start_time = time.time()
    training_history = list(prior_history)
    total_epochs = start_epoch + epochs

    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = time.time()
        model_G.train()
        model_D.train()

        running_loss_G = 0.0
        running_content_loss = 0.0
        running_vgg_loss = 0.0
        running_adv_loss = 0.0
        running_loss_D = 0.0
        running_loss_D_real = 0.0
        running_loss_D_fake = 0.0
        n_d_updates = 0

        print(f"Epoch {epoch+1}/{total_epochs}", flush=True)
        pbar = tqdm(
            train_loader,
            desc="Training",
            file=sys.stdout,
            dynamic_ncols=True,
            leave=False,
        )

        for batch_idx, (lr, hr, _) in enumerate(pbar):
            lr = lr.to(device)
            hr = hr.to(device)

            update_d = (batch_idx % d_interval == 0)

            # ---------------------------------------------
            # 1. Update Discriminator: maximize D(HR) + (1 - D(G(LR)))
            # ---------------------------------------------
            if update_d:
                optimizer_D.zero_grad()

                if scaler_D:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        sr_fake = model_G(lr).detach()
                        real_logits = model_D(hr)
                        fake_logits = model_D(sr_fake)
                        d_loss, d_loss_real, d_loss_fake = calculate_discriminator_loss(
                            real_logits, fake_logits, label_smoothing=label_smoothing
                        )
                    if not torch.isfinite(d_loss):
                        raise FloatingPointError(f"Discriminator loss is non-finite at epoch {epoch+1}, batch {batch_idx}: {d_loss.item()}")
                    scaler_D.scale(d_loss).backward()
                    scaler_D.step(optimizer_D)
                    scaler_D.update()
                else:
                    sr_fake = model_G(lr).detach()
                    real_logits = model_D(hr)
                    fake_logits = model_D(sr_fake)
                    d_loss, d_loss_real, d_loss_fake = calculate_discriminator_loss(
                        real_logits, fake_logits, label_smoothing=label_smoothing
                    )
                    if not torch.isfinite(d_loss):
                        raise FloatingPointError(f"Discriminator loss is non-finite at epoch {epoch+1}, batch {batch_idx}: {d_loss.item()}")
                    d_loss.backward()
                    optimizer_D.step()

                running_loss_D += d_loss.item()
                running_loss_D_real += d_loss_real.item()
                running_loss_D_fake += d_loss_fake.item()
                n_d_updates += 1

            # ---------------------------------------------
            # 2. Update Generator: minimize L_content + L_vgg + L_adv
            # ---------------------------------------------
            optimizer_G.zero_grad()

            # Temporarily freeze Discriminator parameters during Generator backward pass
            # to guarantee D parameters never accumulate gradients from G's objective
            for p in model_D.parameters():
                p.requires_grad = False

            if scaler_G:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    sr_gen = model_G(lr)
                    disc_fake_logits = model_D(sr_gen)
                    g_loss, c_loss, v_loss, a_loss = criterion_G(sr_gen, hr, disc_fake_logits)
                if not torch.isfinite(g_loss):
                    raise FloatingPointError(f"Generator loss is non-finite at epoch {epoch+1}, batch {batch_idx}: {g_loss.item()}")
                scaler_G.scale(g_loss).backward()
                scaler_G.step(optimizer_G)
                scaler_G.update()
            else:
                sr_gen = model_G(lr)
                disc_fake_logits = model_D(sr_gen)
                g_loss, c_loss, v_loss, a_loss = criterion_G(sr_gen, hr, disc_fake_logits)
                if not torch.isfinite(g_loss):
                    raise FloatingPointError(f"Generator loss is non-finite at epoch {epoch+1}, batch {batch_idx}: {g_loss.item()}")
                g_loss.backward()
                optimizer_G.step()

            # Restore Discriminator parameters requires_grad
            for p in model_D.parameters():
                p.requires_grad = True

            running_loss_G += g_loss.item()
            running_content_loss += c_loss.item()
            running_vgg_loss += v_loss.item()
            running_adv_loss += a_loss.item()

            pbar_dict = {"loss_G": f"{g_loss.item():.4f}"}
            if update_d:
                pbar_dict["loss_D"] = f"{d_loss.item():.4f}"
                pbar_dict["D_real"] = f"{d_loss_real.item():.3f}"
                pbar_dict["D_fake"] = f"{d_loss_fake.item():.3f}"
            pbar.set_postfix(pbar_dict)

        n_batches = len(train_loader)
        epoch_loss_G = running_loss_G / n_batches
        epoch_content = running_content_loss / n_batches
        epoch_vgg = running_vgg_loss / n_batches
        epoch_adv_raw = running_adv_loss / n_batches
        epoch_adv_weighted = epoch_adv_raw * adv_weight
        epoch_loss_D = (running_loss_D / n_d_updates) if n_d_updates > 0 else 0.0
        epoch_loss_D_real = (running_loss_D_real / n_d_updates) if n_d_updates > 0 else 0.0
        epoch_loss_D_fake = (running_loss_D_fake / n_d_updates) if n_d_updates > 0 else 0.0
        epoch_time = time.time() - epoch_start_time

        # Verify model parameter health (no non-finite values)
        for p_name, param in model_G.named_parameters():
            if not torch.all(torch.isfinite(param.data)):
                raise FloatingPointError(f"Non-finite value in Generator parameter: {p_name}")
        for p_name, param in model_D.named_parameters():
            if not torch.all(torch.isfinite(param.data)):
                raise FloatingPointError(f"Non-finite value in Discriminator parameter: {p_name}")

        print(f"Training: 100%|{'#'*10}| {n_batches}/{n_batches}", flush=True)
        print(
            f"Loss G: {epoch_loss_G:.6f} (Content: {epoch_content:.6f} | VGG: {epoch_vgg:.6f} | "
            f"Adv Raw: {epoch_adv_raw:.6f} [Weighted: {epoch_adv_weighted:.6f}])\n"
            f"Loss D: {epoch_loss_D:.6f} (Real: {epoch_loss_D_real:.6f} | Fake: {epoch_loss_D_fake:.6f}) [D Updates: {n_d_updates}/{n_batches}] | "
            f"Time: {epoch_time:.1f}s\n",
            flush=True,
        )

        # Validation Phase
        val_metrics = validate_srgan(model_G, val_loader, device, use_amp, show_progress=True)
        val_loss = val_metrics["val_loss"]
        val_psnr = val_metrics["gen_psnr"]
        val_ssim = val_metrics["gen_ssim"]

        is_best = val_psnr > best_val_psnr
        if is_best:
            best_val_psnr = val_psnr

        print(f"Validation: 100%|{'#'*10}| {len(val_loader)}/{len(val_loader)}", flush=True)
        print(f"Val Loss: {val_loss:.6f}", flush=True)
        print(f"Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}\n", flush=True)

        # Save Checkpoint
        latest_file, best_file = save_srgan_checkpoints(
            epoch=epoch,
            model_G=model_G,
            model_D=model_D,
            optimizer_G=optimizer_G,
            optimizer_D=optimizer_D,
            scaler_G=scaler_G,
            scaler_D=scaler_D,
            val_psnr=val_psnr,
            val_ssim=val_ssim,
            val_loss=val_loss,
            is_best=is_best,
            checkpoint_dir=checkpoint_dir,
            prefix=checkpoint_prefix,
        )

        try:
            rel_latest = latest_file.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            rel_latest = str(latest_file)
        print(f"Checkpoint saved: {rel_latest}", flush=True)

        if best_file:
            try:
                rel_best = best_file.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
            except ValueError:
                rel_best = str(best_file)
            print(f"Best model saved: {rel_best}", flush=True)
        print(flush=True)

        epoch_record = {
            "epoch": epoch + 1,
            "loss_G": epoch_loss_G,
            "loss_content": epoch_content,
            "loss_vgg": epoch_vgg,
            "loss_adv_raw": epoch_adv_raw,
            "loss_adv_weighted": epoch_adv_weighted,
            "loss_D": epoch_loss_D,
            "loss_D_real": epoch_loss_D_real,
            "loss_D_fake": epoch_loss_D_fake,
            "d_updates": n_d_updates,
            "g_updates": n_batches,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "epoch_time_seconds": epoch_time,
        }
        training_history.append(epoch_record)

        # Save cumulative training history after each epoch
        hist_out = Path(history_file) if history_file else (checkpoint_dir / f"{checkpoint_prefix}_history.json")
        hist_out.parent.mkdir(parents=True, exist_ok=True)
        with open(hist_out, "w") as f:
            json.dump(training_history, f, indent=2)

    total_time = time.time() - total_start_time
    print(f"SRGAN adversarial training finished in {total_time:.1f}s.", flush=True)
    return {
        "epochs_completed": epochs,
        "total_time_seconds": total_time,
        "history": training_history,
        "best_val_psnr": best_val_psnr,
    }


def main():
    parser = argparse.ArgumentParser(description="SRGAN Adversarial Training")
    parser.add_argument("--epochs", type=int, default=25, help="Number of adversarial epochs to train")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr_g", type=float, default=config.GENERATOR_LR, help="Generator learning rate")
    parser.add_argument("--lr_d", type=float, default=config.DISCRIMINATOR_LR, help="Discriminator learning rate")
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=config.DISCRIMINATOR_LABEL_SMOOTHING,
        help="One-sided label smoothing for real targets (default 0.1 -> target 0.9)",
    )
    parser.add_argument(
        "--d_interval",
        type=int,
        default=config.DISCRIMINATOR_UPDATE_INTERVAL,
        help="Discriminator update interval: 1 D update per d_interval G updates (default 2)",
    )
    parser.add_argument("--content_weight", type=float, default=1.0, help="Content MSE loss weight")
    parser.add_argument("--vgg_weight", type=float, default=config.CONTENT_VGG_WEIGHT, help="VGG perceptual loss weight")
    parser.add_argument("--adv_weight", type=float, default=config.ADVERSARIAL_WEIGHT, help="Adversarial loss weight")
    parser.add_argument(
        "--pretrained_g",
        type=str,
        default="checkpoints/generator_pretrain_best.pth",
        help="Pretrained Generator checkpoint path",
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to SRGAN generator checkpoint to resume from")
    parser.add_argument("--use_vgg", action="store_true", help="Enable pretrained VGG19 perceptual loss")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(config.CHECKPOINTS_DIR),
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--checkpoint_prefix",
        type=str,
        default="srgan",
        help="Prefix for saved checkpoint filenames",
    )
    parser.add_argument(
        "--history_file",
        type=str,
        default=None,
        help="Optional path to save JSON training history",
    )
    parser.add_argument("--smoke_test", action="store_true", help="Run single batch smoke test and exit")
    args = parser.parse_args()

    run_srgan_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        label_smoothing=args.label_smoothing,
        d_interval=args.d_interval,
        content_weight=args.content_weight,
        vgg_weight=args.vgg_weight,
        adv_weight=args.adv_weight,
        pretrained_generator_path=args.pretrained_g,
        resume_path=args.resume,
        pretrained_vgg=args.use_vgg,
        checkpoint_dir=Path(args.checkpoint_dir),
        checkpoint_prefix=args.checkpoint_prefix,
        history_file=args.history_file,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
