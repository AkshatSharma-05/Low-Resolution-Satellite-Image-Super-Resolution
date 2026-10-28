"""
SRGAN Discriminator Architecture.
Progressive downsampling convolutional network with numerically stable unscaled logits output.
Based on Ledig et al., "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" (CVPR 2017).
"""

from typing import Tuple
import torch
import torch.nn as nn


class DiscriminatorBlock(nn.Module):
    """
    Discriminator Convolutional Block:
    Conv2d(3x3) -> (optional BatchNorm2d) -> LeakyReLU(negative_slope=0.2).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=not use_batchnorm)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Discriminator(nn.Module):
    """
    SRGAN Discriminator.
    Progressively downsamples a 256x256 image through 8 convolutional stages,
    producing an unscaled scalar logit per image for stable binary classification.
    
    Input:
        (B, 3, 256, 256) High-Resolution or Super-Resolved image in [-1.0, 1.0].
    Output:
        (B, 1) Raw logit (pass through nn.BCEWithLogitsLoss for numerical stability).
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        c = base_channels
        
        # 8 Convolutional Downsampling Stages:
        # 1. 256x256 -> 256x256 (no BN)
        # 2. 256x256 -> 128x128
        # 3. 128x128 -> 128x128
        # 4. 128x128 -> 64x64
        # 5. 64x64   -> 64x64
        # 6. 64x64   -> 32x32
        # 7. 32x32   -> 32x32
        # 8. 32x32   -> 16x16
        self.features = nn.Sequential(
            DiscriminatorBlock(in_channels, c, stride=1, use_batchnorm=False),
            DiscriminatorBlock(c, c, stride=2, use_batchnorm=True),
            DiscriminatorBlock(c, c * 2, stride=1, use_batchnorm=True),
            DiscriminatorBlock(c * 2, c * 2, stride=2, use_batchnorm=True),
            DiscriminatorBlock(c * 2, c * 4, stride=1, use_batchnorm=True),
            DiscriminatorBlock(c * 4, c * 4, stride=2, use_batchnorm=True),
            DiscriminatorBlock(c * 4, c * 8, stride=1, use_batchnorm=True),
            DiscriminatorBlock(c * 8, c * 8, stride=2, use_batchnorm=True),
        )

        # Spatial pooling to (4, 4) to keep dense layer memory footprint manageable
        # for 6GB VRAM GPUs (512 * 4 * 4 = 8,192 features)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Classification Head: Output raw unscaled logits
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 8 * 4 * 4, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Image tensor (B, 3, 256, 256) in [-1.0, 1.0].
        Returns:
            Logits tensor (B, 1). Positive values indicate real, negative indicate fake.
        """
        feat = self.features(x)
        pooled = self.pool(feat)
        logits = self.classifier(pooled)
        return logits

    def predict_probability(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return predicted probability P(real) in [0.0, 1.0] by applying sigmoid to logits.
        """
        return torch.sigmoid(self.forward(x))


if __name__ == "__main__":
    netD = Discriminator()
    x = torch.randn(4, 3, 256, 256)
    logits = netD(x)
    probs = netD.predict_probability(x)
    print("Discriminator test:")
    print("  Input shape :", x.shape)
    print("  Logits shape:", logits.shape, "values:", logits.squeeze(-1).tolist())
    print("  Probs shape :", probs.shape, "values:", probs.squeeze(-1).tolist())
