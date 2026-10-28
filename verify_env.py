import sys
import torch

def verify():
    print("=" * 60)
    print("         ENVIRONMENT & CUDA VERIFICATION REPORT         ")
    print("=" * 60)
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"PyTorch Version   : {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available    : {cuda_available}")
    
    if not cuda_available:
        print("[FAIL] CUDA is NOT available to PyTorch!")
        return False
        
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    
    print(f"Device Count      : {device_count}")
    print(f"Primary GPU Name  : {device_name}")
    print(f"Compute Capability: {cap[0]}.{cap[1]}")
    
    # Check current VRAM allocated and cached
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Total VRAM        : {total_mem:.2f} GB")
    
    print("-" * 60)
    print("Running a small CUDA tensor operation...")
    
    # Small CUDA tensor operation
    device = torch.device("cuda:0")
    a = torch.randn(256, 256, device=device, dtype=torch.float32)
    b = torch.randn(256, 256, device=device, dtype=torch.float32)
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    expected_shape = (256, 256)
    if c.shape == expected_shape and c.is_cuda:
        print(f"[SUCCESS] Tensor matrix multiplication (256x256) on {device_name} passed.")
        print(f"Tensor output shape: {tuple(c.shape)}, Device: {c.device}, Mean value: {c.mean().item():.4f}")
    else:
        print("[FAIL] Tensor operation did not produce expected result.")
        return False
        
    print("=" * 60)
    print("ALL ENVIRONMENT & CUDA VERIFICATION CHECKS PASSED!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
