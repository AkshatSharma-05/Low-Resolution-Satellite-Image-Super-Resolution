"""
SRResNet Generator Pretraining (Reconstruction Phase).
Trains the Generator on paired LR-HR satellite images using pixel-level MSE or L1 loss.
Provides live unbuffered tqdm progress bars, validation monitoring (Loss, PSNR, SSIM),
periodic checkpointing, and resumability.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout is unbuffered for live streaming in terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import argparse
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
from src.baseline import bicubic_upscale
from src.metrics import calculate_psnr, calculate_ssim


def validate_generator(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Evaluate Generator on validation set.
    Computes validation loss, Generator PSNR/SSIM, and Bicubic baseline PSNR/SSIM for comparison.
    Both images are denormalized to [0, 1] range for standard PSNR/SSIM calculation.
    """
    model.eval()
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
                    sr = model(lr)
                    loss = criterion(sr, hr)
            else:
                sr = model(lr)
                loss = criterion(sr, hr)

            val_losses.append(loss.item())

            # Convert to [0, 1] range for standard image quality metric calculation
            sr_01 = denormalize_hr(sr)
            hr_01 = denormalize_hr(hr)

            # Also compute bicubic baseline on the same batch for direct comparison
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


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: Path = config.CHECKPOINTS_DIR,
) -> Tuple[Path, Optional[Path]]:
    """Save training checkpoint to disk."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "generator_pretrain_latest.pth"
    torch.save(state, latest_path)
    
    best_path = None
    if is_best:
        best_path = checkpoint_dir / "generator_pretrain_best.pth"
        torch.save(state, best_path)
        
    return latest_path, best_path


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Load checkpoint weights, optimizer, and training metadata."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    if scaler and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
    return checkpoint


def run_pretraining(
    epochs: int = 1,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.PRETRAIN_LR,
    loss_type: str = "mse",
    use_amp: bool = config.USE_AMP,
    smoke_test: bool = False,
    resume_checkpoint: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Main routine for Generator pretraining with live progress and clean epoch summaries.
    """
    if device is None:
        device = config.DEVICE

    # Formatted Header
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print("SRResNet Generator Pretraining", flush=True)
    print(f"Device: {device_name} | Batch Size: {batch_size} | Epochs: {epochs} | LR: {lr} | Mixed Precision: {use_amp and device.type == 'cuda'}\n", flush=True)

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

    # 2. Initialize Model, Optimizer, Loss, Scaler
    model = Generator(
        in_channels=3,
        out_channels=3,
        num_features=64,
        num_residual_blocks=16,
        upscale_factor=4,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.L1Loss() if loss_type.lower() == "l1" else nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    start_epoch = 0
    best_val_psnr = -float("inf")

    # Resume from checkpoint if requested
    if resume_checkpoint:
        ckpt_path = Path(resume_checkpoint)
        ckpt_data = load_checkpoint(ckpt_path, model, optimizer, scaler, device)
        start_epoch = ckpt_data["epoch"] + 1
        best_val_psnr = ckpt_data.get("val_psnr", -float("inf"))
        print(f"Resumed from epoch {start_epoch} | Previous Best PSNR: {best_val_psnr:.2f} dB\n", flush=True)

    # 3. Handle Smoke Test
    if smoke_test:
        print("Executing one-batch smoke test...", flush=True)
        model.train()
        lr_sample, hr_sample, _ = next(iter(train_loader))
        lr_sample = lr_sample.to(device)
        hr_sample = hr_sample.to(device)

        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                sr_sample = model(lr_sample)
                loss = criterion(sr_sample, hr_sample)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            sr_sample = model(lr_sample)
            loss = criterion(sr_sample, hr_sample)
            loss.backward()
            optimizer.step()

        print(f"[SMOKE TEST SUCCESS] Single batch loss = {loss.item():.6f}\n", flush=True)
        return {"smoke_test_loss": loss.item()}

    # 4. Training Loop with Live Tqdm Progress
    total_start_time = time.time()
    training_history = []
    total_epochs = start_epoch + epochs

    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0

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

            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    sr = model(lr)
                    loss = criterion(sr, hr)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                sr = model(lr)
                loss = criterion(sr, hr)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_train_loss = running_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time

        print(f"Training: 100%|{'█'*10}| {len(train_loader)}/{len(train_loader)}", flush=True)
        print(f"Loss: {epoch_train_loss:.6f} | Time: {epoch_time:.1f}s\n", flush=True)

        # Validation phase
        val_metrics = validate_generator(model, val_loader, criterion, device, use_amp, show_progress=True)
        val_loss = val_metrics["val_loss"]
        val_psnr = val_metrics["gen_psnr"]
        val_ssim = val_metrics["gen_ssim"]
        bi_psnr = val_metrics["bicubic_psnr"]
        bi_ssim = val_metrics["bicubic_ssim"]

        is_best = val_psnr > best_val_psnr
        if is_best:
            best_val_psnr = val_psnr

        print(f"Validation: 100%|{'█'*10}| {len(val_loader)}/{len(val_loader)}", flush=True)
        print(f"Val Loss: {val_loss:.6f}", flush=True)
        print(f"Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}\n", flush=True)

        # Save Checkpoint
        ckpt_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "train_loss": epoch_train_loss,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "bicubic_psnr": bi_psnr,
            "bicubic_ssim": bi_ssim,
            "config": {
                "batch_size": batch_size,
                "lr": lr,
                "loss_type": loss_type,
                "use_amp": use_amp,
            },
        }
        latest_file, best_file = save_checkpoint(ckpt_state, is_best=is_best)
        print(f"Checkpoint saved: {latest_file.relative_to(PROJECT_ROOT).as_posix()}", flush=True)
        if best_file:
            print(f"Best model saved: {best_file.relative_to(PROJECT_ROOT).as_posix()}", flush=True)
        print(flush=True)

        training_history.append({
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "epoch_time_seconds": epoch_time,
        })

    total_time = time.time() - total_start_time
    print(f"Pretraining run finished in {total_time:.1f}s.", flush=True)
    return {
        "epochs_completed": epochs,
        "total_time_seconds": total_time,
        "history": training_history,
        "best_val_psnr": best_val_psnr,
    }


def main():
    parser = argparse.ArgumentParser(description="SRResNet Generator Pretraining")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.PRETRAIN_LR, help="Learning rate")
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "l1"], help="Loss function")
    parser.add_argument("--smoke_test", action="store_true", help="Run single batch smoke test and exit")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    run_pretraining(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_type=args.loss_type,
        smoke_test=args.smoke_test,
        resume_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
