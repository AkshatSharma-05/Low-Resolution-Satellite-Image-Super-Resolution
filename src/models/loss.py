"""
Loss Functions for SRGAN Super-Resolution.

Includes:
1. Pixel Reconstruction Loss (MSE/L1) for Generator pretraining (SRResNet).
2. VGG19-based Perceptual Loss (content loss in deep feature space).
3. Adversarial Loss (BCEWithLogitsLoss) for GAN training.
4. Composite Generator Loss combining Content, Perceptual, and Adversarial terms.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    """
    VGG-19 Perceptual Content Loss.
    
    Mathematical Formulation:
        L_vgg = (1 / (W * H * C)) * || phi(SR) - phi(HR) ||_2^2
        where phi is the feature map extracted by VGG-19 before the 5th max pooling layer.
        
    Feature Layer Selection:
        Following Ledig et al. (CVPR 2017), we use layer phi_{5,4} (vgg19.features[:35]),
        which represents high-level semantic representations (contours, patterns, structural shapes)
        rather than raw pixel alignments. This encourages perceptual sharpness in satellite imagery
        (e.g., sharp runway lines, distinct building perimeters, river edges) without producing
        the over-smoothed plastic appearance typical of pure MSE loss.
        
    Input Range & Normalization Contract:
        Input images are expected in [-1.0, 1.0] (matching Generator output and HR dataset).
        Inside forward(), images are explicitly mapped to [0.0, 1.0]:
            x_01 = (x + 1.0) / 2.0
        Then normalized with standard ImageNet statistics:
            mean = [0.485, 0.456, 0.406]
            std  = [0.229, 0.224, 0.225]
            
    Parameter Freezing & Gradients:
        All VGG parameters are frozen (requires_grad = False) and the network is in eval() mode.
        Autograd gradients are fully preserved through the generated input tensor back to the Generator.
    """
    def __init__(self, feature_layer: int = 35, pretrained: bool = False):
        super().__init__()
        self.feature_layer = feature_layer
        self.pretrained = pretrained

        # Load VGG-19 architecture.
        # When pretrained=False, initializes architecture without network download.
        if pretrained:
            weights = models.VGG19_Weights.DEFAULT
            vgg19 = models.vgg19(weights=weights)
        else:
            vgg19 = models.vgg19(weights=None)

        # Slice VGG-19 features up to the selected layer (index 35 includes Conv5_4 before MaxPool5)
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:feature_layer])
        self.feature_extractor.eval()

        # Completely freeze VGG parameters to eliminate gradient computation and memory overhead
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # ImageNet normalization parameters registered as buffers
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.criterion = nn.MSELoss()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual MSE loss between VGG feature representations.
        
        Args:
            sr: Super-resolved generated image (B, 3, 256, 256) in [-1.0, 1.0].
            hr: Ground truth high-resolution image (B, 3, 256, 256) in [-1.0, 1.0].
            
        Returns:
            Scalar perceptual loss tensor with gradients attached to sr.
        """
        # Step 1: Map from [-1, 1] to [0, 1]
        sr_01 = torch.clamp((sr + 1.0) / 2.0, 0.0, 1.0)
        hr_01 = torch.clamp((hr + 1.0) / 2.0, 0.0, 1.0)

        # Step 2: Apply ImageNet mean/std normalization
        sr_norm = (sr_01 - self.mean) / self.std
        hr_norm = (hr_01 - self.mean) / self.std

        # Step 3: Extract deep feature maps
        sr_features = self.feature_extractor(sr_norm)
        hr_features = self.feature_extractor(hr_norm)

        # Step 4: MSE loss between feature representations
        loss = self.criterion(sr_features, hr_features)
        return loss


class GeneratorLoss(nn.Module):
    """
    Composite Generator Loss for SRGAN adversarial training.
    
    Mathematical Formulation:
        L_G = w_content * L_content + w_vgg * L_vgg + w_adv * L_adv
        
    Where:
        - L_content = MSE(SR, HR) in image space.
        - L_vgg = MSE(phi(SR), phi(HR)) in VGG feature space.
        - L_adv = BCEWithLogits(D(SR), 1.0) [Non-saturating generator loss].
        
    Args:
        content_weight: Weight for pixel-level MSE loss (default 1.0).
        vgg_weight: Weight for perceptual VGG loss (default 1.0).
        adv_weight: Weight for adversarial loss (default 1e-3).
        pretrained_vgg: Whether to load pretrained VGG weights (default False until confirmed).
    """
    def __init__(
        self,
        content_weight: float = 1.0,
        vgg_weight: float = 1.0,
        adv_weight: float = 1e-3,
        feature_layer: int = 35,
        pretrained_vgg: bool = False,
    ):
        super().__init__()
        self.content_weight = content_weight
        self.vgg_weight = vgg_weight
        self.adv_weight = adv_weight

        self.mse_loss = nn.MSELoss()
        self.vgg_loss = VGGPerceptualLoss(feature_layer=feature_layer, pretrained=pretrained_vgg)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        disc_fake_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute composite loss terms.
        
        Args:
            sr: Super-resolved image (B, 3, 256, 256) in [-1.0, 1.0].
            hr: Ground truth HR image (B, 3, 256, 256) in [-1.0, 1.0].
            disc_fake_logits: Discriminator logits on generated SR images (B, 1).
            
        Returns:
            Tuple: (total_loss, content_loss, vgg_loss, adv_loss)
        """
        # 1. Pixel-level MSE reconstruction loss
        content_loss = self.mse_loss(sr, hr)

        # 2. VGG perceptual loss
        vgg_loss = self.vgg_loss(sr, hr)

        # 3. Adversarial loss: Generator wants Discriminator to predict 1 (Real) for SR
        real_labels = torch.ones_like(disc_fake_logits)
        adv_loss = self.bce_loss(disc_fake_logits, real_labels)

        # 4. Weighted total loss
        total_loss = (
            self.content_weight * content_loss
            + self.vgg_weight * vgg_loss
            + self.adv_weight * adv_loss
        )

        return total_loss, content_loss, vgg_loss, adv_loss


def calculate_discriminator_loss(
    disc_real_logits: torch.Tensor,
    disc_fake_logits: torch.Tensor,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Numerically stable Discriminator loss using BCEWithLogitsLoss.
    
    Mathematical Formulation:
        L_D = 0.5 * [ BCE(D(HR), 1) + BCE(D(SR), 0) ]
        
    Args:
        disc_real_logits: Discriminator logits for real HR images (B, 1).
        disc_fake_logits: Discriminator logits for fake generated SR images (B, 1).
        label_smoothing: Optional real label smoothing factor (e.g. 0.9 instead of 1.0).
        
    Returns:
        Tuple: (d_loss, d_real_loss, d_fake_loss)
    """
    criterion = nn.BCEWithLogitsLoss()
    
    real_target = 1.0 - label_smoothing
    real_labels = torch.full_like(disc_real_logits, real_target)
    fake_labels = torch.zeros_like(disc_fake_logits)

    loss_real = criterion(disc_real_logits, real_labels)
    loss_fake = criterion(disc_fake_logits, fake_labels)

    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss, loss_real, loss_fake
