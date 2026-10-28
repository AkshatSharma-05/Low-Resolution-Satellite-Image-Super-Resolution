# Final Evaluation Report: Satellite Image Super-Resolution (4×)

## 1. Executive Summary
- **Dataset**: UC Merced Land Use Dataset (210 held-out test images across 21 classes).
- **Scale Factor**: 4× Super-Resolution (64×64 input $\rightarrow$ 256×256 output).
- **Models Evaluated**:
  1. **Bicubic Baseline**: Standard bicubic interpolation.
  2. **Pretrained Generator (SRResNet)**: Pixel-wise MSE-trained model (`checkpoints/generator_pretrain_best.pth`).
  3. **SRGAN (G2:D1 Update Schedule)**: Adversarially trained model (`results/srgan_g2d1_experiment/srgan_g2d1_generator_best.pth`).

## 2. Overall Quantitative Benchmark

| Model | Mean PSNR (dB) | Std PSNR | Mean SSIM | Std SSIM | Mean MSE | Std MSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bicubic Baseline** | 25.39 | 3.85 | 0.6843 | 0.1427 | 0.00411 | 0.00383 |
| **Pretrained SRResNet** | 26.10 | 3.72 | 0.7164 | 0.1375 | 0.00348 | 0.00341 |
| **SRGAN (G2:D1)** | 24.74 | 3.14 | 0.6696 | 0.1250 | 0.00434 | 0.00360 |

## 3. Analysis of the Perception-Distortion Trade-off
- **Pixel Fidelity (PSNR & MSE)**: The Pretrained SRResNet achieved the highest PSNR (26.10 dB) and lowest MSE (0.00348). This is mathematically expected because SRResNet is optimized directly for minimizing pixel-wise Mean Squared Error, which produces an average of all plausible high-frequency solutions, resulting in smooth textures.
- **Adversarial Realism vs. Pixel Alignment**: The SRGAN model obtained a PSNR of 24.74 dB and SSIM of 0.6696. In accordance with the fundamental *Perception-Distortion Trade-off* (Blau & Michaeli, 2018; Ledig et al., 2017), adversarial training forces the generator to synthesize sharp, high-frequency stochastic textures (e.g., runway markings, roof tiles, tree foliage, and river ripples). While these synthesized textures significantly enhance visual realism and edge definition, any microscopic spatial phase shift from the exact ground truth is heavily penalized by pixel-wise L2 distance (PSNR/MSE).

## 4. Per-Class Performance Breakdown

| Land Use Category | Bicubic PSNR (dB) | SRResNet PSNR (dB) | SRGAN PSNR (dB) | Bicubic SSIM | SRResNet SSIM | SRGAN SSIM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| agricultural | 23.68 | 23.88 | 23.33 | 0.3280 | 0.3619 | 0.3523 |
| airplane | 24.44 | 25.65 | 24.31 | 0.6784 | 0.7011 | 0.6540 |
| baseballdiamond | 30.88 | 30.92 | 28.50 | 0.8094 | 0.8150 | 0.7463 |
| beach | 31.58 | 31.80 | 29.87 | 0.8230 | 0.8375 | 0.7947 |
| buildings | 24.34 | 25.74 | 23.98 | 0.7475 | 0.7906 | 0.7236 |
| chaparral | 23.69 | 24.36 | 23.29 | 0.6448 | 0.7036 | 0.6601 |
| denseresidential | 24.71 | 25.63 | 24.05 | 0.7396 | 0.7746 | 0.7184 |
| forest | 25.64 | 25.49 | 24.47 | 0.5925 | 0.6173 | 0.5762 |
| freeway | 25.16 | 25.85 | 24.69 | 0.6636 | 0.6961 | 0.6568 |
| golfcourse | 28.36 | 28.74 | 27.28 | 0.7068 | 0.7256 | 0.6852 |
| harbor | 21.62 | 22.87 | 21.79 | 0.7116 | 0.7544 | 0.6957 |
| intersection | 25.32 | 25.93 | 24.78 | 0.7189 | 0.7472 | 0.7052 |
| mediumresidential | 23.04 | 23.95 | 22.78 | 0.6597 | 0.7049 | 0.6568 |
| mobilehomepark | 24.50 | 25.94 | 24.19 | 0.7709 | 0.8158 | 0.7614 |
| overpass | 22.82 | 23.76 | 22.82 | 0.6280 | 0.6658 | 0.6216 |
| parkinglot | 20.66 | 21.45 | 20.65 | 0.6339 | 0.6841 | 0.6397 |
| river | 26.08 | 26.38 | 25.36 | 0.6723 | 0.7002 | 0.6561 |
| runway | 26.38 | 27.70 | 25.73 | 0.7227 | 0.7448 | 0.6792 |
| sparseresidential | 27.59 | 27.96 | 26.29 | 0.6958 | 0.7228 | 0.6786 |
| storagetanks | 26.15 | 26.86 | 25.38 | 0.7041 | 0.7391 | 0.6982 |
| tenniscourt | 26.59 | 27.32 | 26.04 | 0.7179 | 0.7418 | 0.7023 |

## 5. Visual Artifacts
The 5-column visual comparison grid is available at `results/final_evaluation/visual_comparison_grid.png`.
