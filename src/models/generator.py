"""
SRResNet Generator Architecture for 4x Super-Resolution (64x64 -> 256x256).
Based on Ledig et al., "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" (CVPR 2017).
"""

from typing import Optional
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual Block with two 3x3 convolutions, Batch Normalization, and PReLU activation.
    Preserves spatial dimensions via padding=1.
    """
    def __init__(self, channels: int = 64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class PixelShuffleBlock(nn.Module):
    """
    Upsampling block using Sub-Pixel Convolution (PixelShuffle) with factor 2x.
    Maps C -> C * 4 -> PixelShuffle(2) -> C.
    """
    def __init__(self, channels: int = 64, upscale_factor: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Generator(nn.Module):
    """
    SRResNet / SRGAN Generator for 4x super-resolution.
    
    Input:
        (B, 3, 64, 64) Low-Resolution image tensor in [0.0, 1.0] range.
    Output:
        (B, 3, 256, 256) High-Resolution image tensor in [-1.0, 1.0] range (via Tanh).
        
    Args:
        in_channels: Input channels (default 3 for RGB).
        out_channels: Output channels (default 3 for RGB).
        num_features: Intermediate feature channels (default 64).
        num_residual_blocks: Number of B residual blocks (default 16).
        upscale_factor: Total upscale factor (default 4 = 2 stages of 2x).
        input_in_01: If True, maps [0, 1] input to [-1, 1] internally for zero-centered feature extraction.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_residual_blocks: int = 16,
        upscale_factor: int = 4,
        input_in_01: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.num_residual_blocks = num_residual_blocks
        self.input_in_01 = input_in_01

        # 1. Initial 9x9 Convolution + PReLU
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
        )

        # 2. Body of B Residual Blocks
        res_blocks = [ResidualBlock(num_features) for _ in range(num_residual_blocks)]
        self.residual_blocks = nn.Sequential(*res_blocks)

        # 3. Post-residual Conv + BatchNorm
        self.post_residual_conv = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features),
        )

        # 4. Upsampling stages (log2(4) = 2 stages for 4x)
        num_upsample_stages = int(torch.log2(torch.tensor(float(upscale_factor))).item())
        upsample_layers = [PixelShuffleBlock(num_features, upscale_factor=2) for _ in range(num_upsample_stages)]
        self.upsample = nn.Sequential(*upsample_layers)

        # 5. Final Reconstruction 9x9 Convolution + Tanh (outputs in [-1, 1])
        self.final_conv = nn.Sequential(
            nn.Conv2d(num_features, out_channels, kernel_size=9, stride=1, padding=4),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Low-resolution tensor (B, 3, 64, 64) in [0.0, 1.0].
        Returns:
            Super-resolved tensor (B, 3, 256, 256) in [-1.0, 1.0].
        """
        # Explicit input range handling: zero-center to [-1, 1] if given [0, 1]
        if self.input_in_01:
            x = x * 2.0 - 1.0

        initial = self.initial_conv(x)
        res_out = self.residual_blocks(initial)
        res_out = self.post_residual_conv(res_out)
        
        # Skip connection around the entire residual trunk
        features = initial + res_out
        
        # 4x PixelShuffle upsampling: (B, 64, 64, 64) -> (B, 64, 256, 256)
        upsampled = self.upsample(features)
        
        # Final reconstruction: (B, 3, 256, 256) in [-1, 1]
        out = self.final_conv(upsampled)
        return out


if __name__ == "__main__":
    netG = Generator()
    x = torch.rand(4, 3, 64, 64)
    y = netG(x)
    print("Generator test:")
    print("  Input shape :", x.shape, "range:", [x.min().item(), x.max().item()])
    print("  Output shape:", y.shape, "range:", [y.min().item(), y.max().item()])
