"""
Streamlit Web Application for Low-Resolution Satellite Image Super-Resolution (4×).
Supports UC Merced Land Use dataset samples and arbitrary user image uploads.
Compares Bicubic Interpolation, Pretrained SRResNet, and SRGAN models side-by-side.
"""

from pathlib import Path
import sys
import io

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
import torch
import streamlit as st

from src import config
from src.inference import (
    validate_and_preprocess_image,
    run_super_resolution,
    compute_image_metrics,
)
from src.evaluate_final import load_generator_checkpoint


st.set_page_config(
    page_title="Satellite Image Super-Resolution (SRGAN 4×)",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_models():
    """Load and cache Pretrained SRResNet and SRGAN models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pretrain_ckpt = config.CHECKPOINTS_DIR / "generator_pretrain_best.pth"
    srgan_ckpt = config.RESULTS_DIR / "srgan_g2d1_experiment" / "srgan_g2d1_generator_best.pth"
    
    model_pretrain = load_generator_checkpoint(pretrain_ckpt, device)
    model_srgan = load_generator_checkpoint(srgan_ckpt, device)
    
    return model_pretrain, model_srgan, device


def get_sample_images():
    """Discover a representative set of dataset images for quick user testing."""
    sample_dir = config.RAW_IMAGES_DIR
    categories = ["airplane", "buildings", "freeway", "harbor", "river", "runway", "storagetanks", "agricultural"]
    samples = {}
    
    if sample_dir.exists():
        for cat in categories:
            cat_dir = sample_dir / cat
            if cat_dir.exists():
                imgs = sorted([p for p in cat_dir.iterdir() if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg"}])
                if imgs:
                    samples[cat.capitalize()] = imgs[0]
    return samples


def image_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    """Convert PIL image to downloadable bytes buffer."""
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


# Main Application
def main():
    st.title("🛰️ Satellite Image 4× Super-Resolution")
    st.markdown(
        "Transform **64×64 Low-Resolution** satellite imagery into **256×256 High-Resolution** "
        "using deep learning. Benchmark **Bicubic Interpolation**, **Pretrained SRResNet**, and **SRGAN**."
    )
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration & Models")
    
    try:
        model_pretrain, model_srgan, device = load_models()
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        st.sidebar.success(f"**Device**: {device_name}")
        st.sidebar.info(
            f"**Pretrained SRResNet**: `checkpoints/generator_pretrain_best.pth`\n\n"
            f"**SRGAN (G2:D1)**: `srgan_g2d1_generator_best.pth`"
        )
    except Exception as e:
        st.sidebar.error(f"Error loading models: {e}")
        st.error(f"Failed to load checkpoints: {e}")
        return

    st.sidebar.markdown("---")
    st.sidebar.header("📁 Input Selection")
    input_mode = st.sidebar.radio(
        "Choose image source:",
        ("Pre-loaded Dataset Sample", "Upload Custom Image"),
    )
    
    input_img = None
    input_source_name = ""
    
    if input_mode == "Pre-loaded Dataset Sample":
        samples = get_sample_images()
        if samples:
            selected_cat = st.sidebar.selectbox("Select Land Use Category:", list(samples.keys()))
            selected_path = samples[selected_cat]
            input_img = Image.open(selected_path)
            input_source_name = f"Dataset Sample: {selected_cat} ({selected_path.name})"
        else:
            st.sidebar.warning("No dataset samples found in raw directory.")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload Satellite Image (.tif, .png, .jpg, .jpeg)",
            type=["tif", "tiff", "png", "jpg", "jpeg"],
        )
        if uploaded_file is not None:
            try:
                input_img = Image.open(uploaded_file)
                input_source_name = f"Uploaded File: {uploaded_file.name}"
            except Exception as e:
                st.error(f"Failed to open uploaded image: {e}")

    if input_img is None:
        st.info("👈 Please select a pre-loaded dataset sample or upload an image from the sidebar to begin.")
        
        # Display static evaluation overview
        st.subheader("📊 Held-Out Test Split Benchmark Overview")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric(label="Bicubic Baseline", value="25.39 dB", delta="SSIM: 0.6843")
        with col_m2:
            st.metric(label="Pretrained SRResNet", value="26.10 dB", delta="+0.71 dB (SSIM: 0.7164)")
        with col_m3:
            st.metric(label="SRGAN (G2:D1)", value="24.74 dB", delta="Perceptual Textures")
            
        final_grid_path = config.RESULTS_DIR / "final_evaluation" / "visual_comparison_grid.png"
        if final_grid_path.exists():
            st.image(str(final_grid_path), caption="Multi-Class Super-Resolution Test Set Comparison Grid", use_container_width=True)
        return

    # Image Preprocessing & Validation
    lr_img, hr_ref, status_note = validate_and_preprocess_image(input_img)
    
    st.markdown(f"**Active Image**: `{input_source_name}`")
    st.info(f"ℹ️ **Status**: {status_note}")
    
    # Run Inference
    with st.spinner("Executing 4× super-resolution inference on CUDA/CPU..."):
        results = run_super_resolution(lr_img, model_pretrain, model_srgan, device)
        
    sr_bicubic = results["bicubic"]
    sr_pretrain = results["pretrain"]
    sr_srgan = results["srgan"]
    
    # Calculate metrics if HR reference is available
    metrics = {}
    if hr_ref is not None:
        metrics["bicubic"] = compute_image_metrics(hr_ref, sr_bicubic)
        metrics["pretrain"] = compute_image_metrics(hr_ref, sr_pretrain)
        metrics["srgan"] = compute_image_metrics(hr_ref, sr_srgan)

    # Visual Comparison Display
    st.subheader("🔍 Side-by-Side Visual Comparison")
    
    # Low-res zoomed for visual clarity
    lr_zoomed = lr_img.resize((256, 256), resample=Image.Resampling.NEAREST)
    
    if hr_ref is not None:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.image(lr_zoomed, caption="1. LR Input (64×64)\n[Nearest Zoom]", use_container_width=True)
        with c2:
            caption = f"2. Bicubic Baseline\n{metrics['bicubic']['psnr']:.2f} dB | {metrics['bicubic']['ssim']:.4f}"
            st.image(sr_bicubic, caption=caption, use_container_width=True)
        with c3:
            caption = f"3. SRResNet (MSE)\n{metrics['pretrain']['psnr']:.2f} dB | {metrics['pretrain']['ssim']:.4f}"
            st.image(sr_pretrain, caption=caption, use_container_width=True)
        with c4:
            caption = f"4. SRGAN (Adversarial)\n{metrics['srgan']['psnr']:.2f} dB | {metrics['srgan']['ssim']:.4f}"
            st.image(sr_srgan, caption=caption, use_container_width=True)
        with c5:
            st.image(hr_ref, caption="5. Ground Truth HR\n(256×256 Reference)", use_container_width=True)
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.image(lr_zoomed, caption="1. LR Input (64×64)\n[Nearest Zoom]", use_container_width=True)
        with c2:
            st.image(sr_bicubic, caption="2. Bicubic Baseline (256×256)", use_container_width=True)
        with c3:
            st.image(sr_pretrain, caption="3. SRResNet (MSE Pretrained)", use_container_width=True)
        with c4:
            st.image(sr_srgan, caption="4. SRGAN (High-Frequency Details)", use_container_width=True)

    # Quantitative Comparison Table if HR ground truth available
    if hr_ref is not None:
        st.subheader("📈 Quantitative Reconstruction Quality")
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            st.metric(
                label="Bicubic Interpolation",
                value=f"{metrics['bicubic']['psnr']:.2f} dB",
                delta=f"SSIM: {metrics['bicubic']['ssim']:.4f} | MSE: {metrics['bicubic']['mse']:.5f}",
                delta_color="off",
            )
        with m_col2:
            diff_psnr = metrics['pretrain']['psnr'] - metrics['bicubic']['psnr']
            st.metric(
                label="Pretrained SRResNet (Pixel MSE)",
                value=f"{metrics['pretrain']['psnr']:.2f} dB",
                delta=f"+{diff_psnr:.2f} dB vs Bicubic | SSIM: {metrics['pretrain']['ssim']:.4f}",
            )
        with m_col3:
            st.metric(
                label="SRGAN (Perceptual + Adversarial)",
                value=f"{metrics['srgan']['psnr']:.2f} dB",
                delta=f"SSIM: {metrics['srgan']['ssim']:.4f} | Sharp Edges",
                delta_color="off",
            )

    # Downloads
    st.subheader("💾 Export Super-Resolved Images")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            label="⬇️ Download Bicubic Output",
            data=image_to_bytes(sr_bicubic),
            file_name="super_resolution_bicubic.png",
            mime="image/png",
        )
    with d2:
        st.download_button(
            label="⬇️ Download SRResNet Output",
            data=image_to_bytes(sr_pretrain),
            file_name="super_resolution_srresnet.png",
            mime="image/png",
        )
    with d3:
        st.download_button(
            label="⬇️ Download SRGAN Output",
            data=image_to_bytes(sr_srgan),
            file_name="super_resolution_srgan.png",
            mime="image/png",
        )

    # Technical Expander on Perception-Distortion Trade-off
    with st.expander("📚 Understanding the Perception-Distortion Trade-off"):
        st.markdown("""
        ### Why does SRGAN have lower PSNR than SRResNet?
        
        In Super-Resolution literature (e.g., *Blau & Michaeli, "The Perception-Distortion Tradeoff", CVPR 2018* and *Ledig et al., CVPR 2017*):
        
        1. **Pixel-wise MSE Optimization (SRResNet)**:
           - Models trained purely with $L_2$ pixel loss (MSE) find the **mathematical mean** of all possible high-frequency textures that could correspond to the low-resolution input.
           - While this mathematically maximizes Peak Signal-to-Noise Ratio (PSNR), averaging out textures causes smoothing and blurry edges on fine satellite features (roads, roofs, airplane wings).
           
        2. **Adversarial Optimization (SRGAN)**:
           - SRGAN uses a Discriminator and a VGG perceptual loss to drive the generator into the natural image manifold.
           - It synthesizes realistic, high-frequency stochastic details (e.g. runway markings, rooftop textures, tree canopy).
           - Even if a synthesized texture is visually realistic, a microscopic phase shift of a few pixels from the reference ground truth is heavily penalized by pixel-by-pixel Euclidean distance ($L_2$ error), lowering PSNR while significantly boosting perceived sharpness and structural contrast.
        """)


if __name__ == "__main__":
    main()
