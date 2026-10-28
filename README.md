# Low-Resolution Satellite Image Super-Resolution

## 1. Project Overview
This project implements 4× single-image super-resolution for satellite remote sensing imagery. The system upscales 64×64 low-resolution (LR) satellite images to 256×256 high-resolution (HR) outputs, comparing classical interpolation against deep learning architectures (SRResNet and SRGAN).

## 2. Key Features
- **Dataset Pipeline**: Automated bicubic degradation (256×256 to 64×64) and class-stratified splits.
- **Bicubic Baseline**: Non-learned interpolation benchmark for 4× magnification.
- **SRResNet Generator**: 16-residual-block convolutional network optimized with pixel-wise MSE loss.
- **SRGAN Adversarial Training**: Discriminator network with VGG-19 perceptual loss and a stabilized 2:1 generator-to-discriminator update schedule.
- **Quantitative Benchmark**: Rigorous held-out test evaluation using PSNR, SSIM, and MSE.
- **Interactive Streamlit App**: Browser-based interface for side-by-side comparison, image validation, and metric inspection.

## 3. Dataset
The project benchmarks on the **UC Merced Land Use Dataset**, consisting of 2,100 RGB aerial images across 21 land-use categories (100 images per class, originally 256×256 pixels).
- **Split Distribution**: 1,680 training (80%), 210 validation (10%), and 210 held-out test images (10%).
- **LR Generation**: 64×64 LR images are synthesized on the fly via bicubic downsampling (`PIL.Image.Resampling.BICUBIC`).
- **Data Location**: The raw imagery must be placed in `data/raw/Images/<category>/`. Split indices are preserved in `data/splits/splits_manifest.json`. Raw images are not bundled with Git and must be obtained separately if not already present.

## 4. Models and Methodology
1. **Bicubic Interpolation**: Baseline 4× spatial interpolation using PyTorch `F.interpolate` with `align_corners=False`.
2. **SRResNet Generator**: Deep residual network featuring 16 residual blocks with skip connections and two `PixelShuffle(2)` sub-pixel convolutional layers. Pretrained with $L_2$ pixel loss (MSE).
3. **SRGAN (Adversarial Setup)**:
   - **Discriminator**: VGG-style binary classifier producing unnormalized logits for BCE with logits loss.
   - **Loss Formulation**: Weighted composite loss consisting of Content MSE ($1.0$), VGG-19 high-level feature loss ($0.1$ at $\text{Conv}_{5,4}$), and adversarial loss ($0.001$).
   - **Stabilization**: One-sided real-label smoothing ($0.9$ real, $0.0$ fake) and a 2:1 update ratio (two generator updates per discriminator step with discriminator parameter freezing during generator backpropagation).

## 5. Results
Evaluation on the untouched held-out test split (210 samples across 21 classes) yielded the following metrics:

| Model | PSNR (dB) | SSIM | MSE |
|---|---:|---:|---:|
| Bicubic | 25.39 | 0.6843 | 0.00411 |
| Pretrained SRResNet | **26.10** | **0.7164** | **0.00348** |
| SRGAN (2:1 updates) | 24.74 | 0.6696 | 0.00434 |

**Pretrained SRResNet is the best-performing model** across all three reported test metrics, achieving the highest PSNR and lowest MSE. While SRResNet optimizes directly for pixel alignment, SRGAN synthesizes high-frequency textures and sharp edge transitions that are visually sharper but penalized by pixel-wise distance metrics (the classical Perception-Distortion Trade-off).

## 6. Project Structure
```text
Low-Resolution Satellite Image Super-Resolution/
├── app.py                      # Interactive Streamlit application
├── requirements.txt            # Direct Python dependencies
├── verify_env.py               # Environment and CUDA verification script
├── checkpoints/                # Saved model weights
│   └── generator_pretrain_best.pth
├── data/
│   ├── raw/Images/             # Category folders (e.g., airplane, runway)
│   └── splits/                 # Train/val/test split manifests
├── results/
│   ├── final_evaluation/       # Final benchmark metrics, report, and visual grid
│   └── srgan_g2d1_experiment/  # Best SRGAN model weights and logs
├── src/
│   ├── baseline.py             # Bicubic baseline implementation
│   ├── config.py               # Hyperparameters and paths
│   ├── dataset.py              # PyTorch Dataset and preprocessing
│   ├── evaluate_final.py       # Comprehensive test evaluation pipeline
│   ├── inference.py            # Preprocessing, validation, and inference helpers
│   ├── metrics.py              # PSNR, SSIM, and MSE functions
│   ├── train_pretrain.py       # Generator pretraining pipeline
│   ├── train_srgan.py          # SRGAN adversarial training pipeline
│   └── models/
│       ├── discriminator.py    # VGG-style patch discriminator
│       ├── generator.py        # SRResNet / SRGAN generator
│       └── loss.py             # Content, VGG-19, and adversarial losses
└── tests/                      # 34 automated unit and integration tests
```

## 7. Installation and Setup
On Windows PowerShell:

```powershell
# 1. Clone repository and navigate to root
cd "Low-Resolution Satellite Image Super-Resolution"

# 2. Create and activate a virtual environment (Python 3.11 recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install PyTorch with CUDA 12.4 support (for GPU acceleration)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 4. Install remaining dependencies
pip install -r requirements.txt
```

## 8. How to Run
All commands should be executed from the project root using the virtual environment:

- **Launch Interactive Streamlit Web App**:
  ```powershell
  .\.venv\Scripts\python.exe -m streamlit run app.py
  ```
  Open your web browser at `http://localhost:8501`.

- **Run Automated Test Suite**:
  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests -v
  ```

- **Run Final Held-Out Test Evaluation**:
  ```powershell
  .\.venv\Scripts\python.exe src/evaluate_final.py
  ```

## 9. Output Artifacts
Benchmark artifacts are located in `results/final_evaluation/`:
- `final_metrics_summary.json`: Detailed per-class and global metrics across all 210 test images.
- `final_evaluation_report.md`: Markdown summary detailing the quantitative results and per-category breakdown.
- `visual_comparison_grid.png`: High-resolution 5-column comparison grid covering diverse land-use categories.

## 10. Limitations
- **Dataset Scale**: Benchmarked on 2,100 aerial scenes; performance on ultra-high-resolution satellite sensors (e.g., WorldView) or unseen atmospheric conditions may require fine-tuning.
- **Computational Cost**: Adversarial training with perceptual loss requires substantial VRAM and compute compared to standard interpolation.
- **Metric Divergence**: SRGAN did not outperform the pretrained SRResNet on pixel-distortion metrics (PSNR, SSIM, MSE), as adversarial textures do not match ground truth on a pixel-by-pixel basis.
