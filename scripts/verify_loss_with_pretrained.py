import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.loss import VGGPerceptualLoss, GeneratorLoss

print("--- Verifying VGGPerceptualLoss with Pretrained Weights ---")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vgg_loss = VGGPerceptualLoss(feature_layer=35, pretrained=True).to(device)

# 1. Confirm all VGG parameters have requires_grad = False
all_frozen = all(not p.requires_grad for p in vgg_loss.parameters())
print(f"1. All VGG parameters frozen (requires_grad=False): {all_frozen}")
assert all_frozen, "Error: VGG parameters are not frozen!"

# 2. Confirm feature extractor is in eval mode
is_eval = not vgg_loss.feature_extractor.training
print(f"2. Feature extractor in eval mode: {is_eval}")
assert is_eval, "Error: Feature extractor is not in eval mode!"

# 3. Test forward pass with inputs in [-1, 1]
sr = torch.randn(2, 3, 256, 256, device=device, requires_grad=True)
hr = torch.empty(2, 3, 256, 256, device=device).uniform_(-1.0, 1.0)

loss = vgg_loss(sr, hr)
print(f"3. Forward pass loss value: {loss.item():.6f} (finite: {torch.isfinite(loss).item()})")
assert torch.isfinite(loss), "Error: VGG loss is not finite!"

# 4. Confirm gradient propagation back to sr
loss.backward()
grad_ok = sr.grad is not None and torch.isfinite(sr.grad).all() and sr.grad.abs().sum().item() > 0
print(f"4. Gradients flow back to generated image (sr.grad valid): {grad_ok}")
assert grad_ok, "Error: Gradients did not flow back to sr!"

# 5. Confirm VGG parameters received NO gradients
vgg_grads = [p.grad for p in vgg_loss.parameters() if p.grad is not None]
print(f"5. VGG parameters with gradients: {len(vgg_grads)} (expected 0)")
assert len(vgg_grads) == 0, "Error: VGG parameters received gradients!"

print("\n--- ALL PERCEPTUAL LOSS CHECKS WITH PRETRAINED WEIGHTS PASSED ---")
