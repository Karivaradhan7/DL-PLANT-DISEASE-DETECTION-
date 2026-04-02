"""
DL Plant Disease Detection - Streamlit Application
Full integration of all models: CNN, MLP, ResNet50, MobileNetV2, LSTM, GRU, Autoencoder, GAN
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import yaml
from pathlib import Path

from src.models.classifiers import CNNClassifier, MLPClassifier
from src.models.temporal import PretrainedExtractor, SequenceModel
from src.models.autoencoder import ConvAutoencoder
from src.models.dcgan import DCGANGenerator, DCGANDiscriminator
from src.utils.misc import get_device, set_seed
from src.utils.class_names import CLASS_NAMES, get_class_display_name


# ========================================
# PAGE CONFIG & STYLING
# ========================================
st.set_page_config(
    page_title="DL-Plant-Disease-System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title { font-size: 3em; color: #2ecc71; font-weight: bold; }
    .section-title { font-size: 2em; color: #3498db; font-weight: bold; margin-top: 20px; }
    .metric-box { background-color: #ecf0f1; padding: 10px; border-radius: 5px; }
    .success-box { background-color: #d5f4e6; padding: 15px; border-radius: 5px; border-left: 5px solid #27ae60; }
    .error-box { background-color: #fadbd8; padding: 15px; border-radius: 5px; border-left: 5px solid #e74c3c; }
    </style>
""", unsafe_allow_html=True)


# ========================================
# LOAD CONFIG
# ========================================
@st.cache_resource
def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

cfg = load_config()
set_seed(cfg['seed'])

# Setup base directory for model paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device('cpu')

# ========================================
# LOAD CLASSES DYNAMICALLY
# ========================================
@st.cache_resource
def load_classes():
    """Load classes from trained model directory"""
    classes_path = os.path.join(BASE_DIR, "outputs/models/classes.json")
    if os.path.exists(classes_path):
        import json
        with open(classes_path) as f:
            classes = json.load(f)
        st.write(f"📊 Loaded {len(classes)} classes from dataset")
        return classes
    else:
        # Fallback to hardcoded classes if no trained models
        st.warning("⚠️ No trained classes found, using fallback classes")
        return ["class_a", "class_b"]  # Default for synthetic dataset

CLASSES = load_classes()
NUM_CLASSES = len(CLASSES)

# ========================================
# AUTO RETRAINING
# ========================================
@st.cache_resource
def check_and_retrain_models():
    """Check if models exist, retrain if necessary"""
    cnn_model_path = os.path.join(BASE_DIR, "outputs/models/review1/cnn.pth")
    
    if not os.path.exists(cnn_model_path):
        st.warning("🔄 Models not found. Starting automatic training...")
        
        # Run training
        import subprocess
        try:
            result = subprocess.run(
                ["python", "train.py", "--review", "1"], 
                cwd=BASE_DIR,
                capture_output=True, 
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode == 0:
                st.success("✅ Training completed successfully!")
                # Reload classes after training
                global CLASSES, NUM_CLASSES
                CLASSES = load_classes()
                NUM_CLASSES = len(CLASSES)
                return True
            else:
                st.error(f"❌ Training failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            st.error("❌ Training timed out after 10 minutes")
            return False
        except Exception as e:
            st.error(f"❌ Error during training: {e}")
            return False
    
    return True

# Check models on app start
models_ready = check_and_retrain_models()
if not models_ready:
    st.error("Failed to initialize models. Please check the training process.")


# ========================================
# HELPER FUNCTIONS
# ========================================
def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


@st.cache_resource
def load_cnn_model(device):
    """Load CNN classifier model"""
    models_to_try = [
        os.path.join(BASE_DIR, "outputs/models/review1/cnn.pth"),
        os.path.join(BASE_DIR, "outputs/results/review1/cnn_model.pt"),
    ]
    
    model = CNNClassifier(num_classes=NUM_CLASSES)
    
    for model_path in models_to_try:
        if os.path.exists(model_path):
            try:
                st.write(f"📍 Loading CNN model from: {model_path}")
                state_dict = torch.load(model_path, map_location=torch.device("cpu"))
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                st.success(f"✅ CNN model loaded successfully")
                return model
            except Exception as e:
                st.error(f"❌ Failed to load CNN from {model_path}: {e}")
                continue
    
    st.error(f"❌ CNN model not found at any location: {models_to_try}")
    return None


@st.cache_resource
def load_mlp_model(device):
    """Load MLP classifier model"""
    models_to_try = [
        os.path.join(BASE_DIR, "outputs/models/review1/mlp.pth"),
        os.path.join(BASE_DIR, "outputs/results/review1/mlp_model.pt"),
    ]
    
    model = MLPClassifier(
        num_features=3*128*128, 
        hidden_size=256, 
        num_classes=NUM_CLASSES
    )
    
    for model_path in models_to_try:
        if os.path.exists(model_path):
            try:
                st.write(f"📍 Loading MLP model from: {model_path}")
                state_dict = torch.load(model_path, map_location=torch.device("cpu"))
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                st.success(f"✅ MLP model loaded successfully")
                return model
            except Exception as e:
                st.error(f"❌ Failed to load MLP from {model_path}: {e}")
                continue
    
    st.error(f"❌ MLP model not found at any location: {models_to_try}")
    return None


@st.cache_resource
def load_resnet50_model(device):
    """Load ResNet50 pretrained feature extractor"""
    model = PretrainedExtractor(model_name='resnet50', pretrained=True)
    model.to(device)
    model.eval()
    return model


@st.cache_resource
def load_mobilenetv2_model(device):
    """Load MobileNetV2 pretrained feature extractor"""
    model = PretrainedExtractor(model_name='mobilenet_v2', pretrained=True)
    model.to(device)
    model.eval()
    return model


@st.cache_resource
def load_lstm_model(device):
    """Load LSTM sequence model"""
    models_to_try = [
        os.path.join(BASE_DIR, "outputs/models/review2/resnet50_lstm.pth"),
        os.path.join(BASE_DIR, "outputs/results/review2/resnet50_LSTM_attn_False.pt"),
    ]
    
    model = SequenceModel(
        input_size=2048, 
        hidden_size=128, 
        num_classes=NUM_CLASSES, 
        rnn_type='LSTM', 
        use_attention=False
    )
    
    for model_path in models_to_try:
        if os.path.exists(model_path):
            try:
                st.write(f"📍 Loading LSTM model from: {model_path}")
                state_dict = torch.load(model_path, map_location=torch.device("cpu"))
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                st.success(f"✅ LSTM model loaded successfully")
                return model
            except Exception as e:
                st.error(f"❌ Failed to load LSTM from {model_path}: {e}")
                continue
    
    return None


@st.cache_resource
def load_gru_model(device):
    """Load GRU sequence model"""
    models_to_try = [
        os.path.join(BASE_DIR, "outputs/models/review2/resnet50_gru.pth"),
        os.path.join(BASE_DIR, "outputs/results/review2/resnet50_GRU_attn_False.pt"),
    ]
    
    model = SequenceModel(
        input_size=2048, 
        hidden_size=128, 
        num_classes=NUM_CLASSES, 
        rnn_type='GRU', 
        use_attention=False
    )
    
    for model_path in models_to_try:
        if os.path.exists(model_path):
            try:
                st.write(f"📍 Loading GRU model from: {model_path}")
                state_dict = torch.load(model_path, map_location=torch.device("cpu"))
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                st.success(f"✅ GRU model loaded successfully")
                return model
            except Exception as e:
                st.error(f"❌ Failed to load GRU from {model_path}: {e}")
                continue
    
    return None


@st.cache_resource
def load_autoencoder_model(device):
    """Load Autoencoder model"""
    models_to_try = [
        os.path.join(BASE_DIR, "outputs/models/review3/autoencoder.pth"),
        os.path.join(BASE_DIR, "outputs/results/review3/autoencoder.pt"),
    ]
    
    model = ConvAutoencoder(latent_dim=64)
    
    for model_path in models_to_try:
        if os.path.exists(model_path):
            try:
                st.write(f"📍 Loading Autoencoder model from: {model_path}")
                state_dict = torch.load(model_path, map_location=torch.device("cpu"))
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                st.success(f"✅ Autoencoder model loaded successfully")
                return model
            except Exception as e:
                st.error(f"❌ Failed to load Autoencoder from {model_path}: {e}")
                continue
    
    return None


@st.cache_resource
def load_gan_generator(device):
    """Load GAN generator model"""
    models_to_try = [
        os.path.join(BASE_DIR, "outputs/models/review3/gan_generator.pth"),
        os.path.join(BASE_DIR, "outputs/results/review3/gan_generator.pt"),
    ]
    
    model = DCGANGenerator(z_dim=100)
    
    for model_path in models_to_try:
        if os.path.exists(model_path):
            try:
                st.write(f"📍 Loading GAN Generator model from: {model_path}")
                state_dict = torch.load(model_path, map_location=torch.device("cpu"))
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                st.success(f"✅ GAN Generator model loaded successfully")
                return model
            except Exception as e:
                st.error(f"❌ Failed to load GAN Generator from {model_path}: {e}")
                continue
    
    return None


def preprocess_image(image, image_size=128):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)


def predict_classification(model, image_tensor, device):
    """Get predictions from classification model"""
    with torch.no_grad():
        output = model(image_tensor.to(device))
        probs = torch.softmax(output, dim=1)
        
    return probs[0].cpu().numpy()


def plot_predictions_bar(probs, title="Class Probabilities"):
    """Plot confidence distribution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    top_k = min(5, len(probs))
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_probs = probs[top_indices]
    top_names = [get_class_display_name(CLASSES[i]) for i in top_indices]
    
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(top_names))]
    ax.barh(top_names, top_probs, color=colors)
    ax.set_xlabel('Probability')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    
    for i, v in enumerate(top_probs):
        ax.text(v + 0.02, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    return fig


# ========================================
# SIDEBAR CONFIG
# ========================================
with st.sidebar:
    st.title("🎛️ Configuration")
    
    model_type = st.selectbox(
        "Select Model Type",
        ["CNN", "MLP", "ResNet50", "MobileNetV2", "LSTM", "GRU", "Autoencoder", "GAN"],
        help="Choose the model for inference"
    )
    
    device_choice = st.radio("Select Device", ["CPU", "GPU"])
    use_gpu = device_choice == "GPU" and torch.cuda.is_available()
    device = get_device(use_gpu)
    
    st.info(f"Device: {device.type.upper()}" + (" (CUDA)" if use_gpu else ""))
    
    st.markdown("---")
    st.markdown(f"### 📊 Class Mapping ({len(CLASSES)} Classes)")
    
    with st.expander("View All Classes"):
        for i, cls in enumerate(CLASSES):
            st.text(f"{i}: {get_class_display_name(cls)}")


# ========================================
# MAIN TABS
# ========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 Classification",
    "🔗 Feature Extraction",
    "🎨 Generative",
    "ℹ️ About"
])


# ========================================
# TAB 1: CLASSIFICATION MODELS
# ========================================
with tab1:
    st.markdown('<h2 class="section-title">Classification Models</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'], key='tab1_upload')
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', width="content")
        else:
            st.info("📌 Upload a plant disease image to get started")
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        if uploaded_file is not None:
            with st.spinner('🔄 Processing...'):
                img_tensor = preprocess_image(image, cfg['image_size'])
                
                # Load appropriate model
                if model_type == "CNN":
                    model = load_cnn_model(device)
                    model_name = "CNN Classifier"
                elif model_type == "MLP":
                    model = load_mlp_model(device)
                    model_name = "MLP Classifier"
                else:
                    st.error("⚠️ Select CNN or MLP for classification")
                    model = None
                
                if model is None:
                    st.markdown('<div class="error-box">', unsafe_allow_html=True)
                    st.error("❌ Model not found!")
                    st.info("Train models first:")
                    st.code("python train.py --review 1")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    probs = predict_classification(model, img_tensor, device)
                    pred_idx = np.argmax(probs)
                    pred_class = CLASSES[pred_idx]
                    confidence = probs[pred_idx] * 100
                    
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.success(f"🌿 **{get_class_display_name(pred_class)}**")
                    st.metric("Confidence", f"{confidence:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.write(f"**Model Type:** {model_name}")
                    
                    # Top-3 Predictions
                    st.markdown("---")
                    st.subheader("🏆 Top Predictions")
                    
                    top3_indices = np.argsort(probs)[::-1][:3]
                    
                    for rank, idx in enumerate(top3_indices, 1):
                        class_name = CLASSES[idx]
                        score = probs[idx] * 100
                        st.write(f"**#{rank}** {get_class_display_name(class_name)} - **{score:.2f}%**")
                    
                    # Confidence distribution chart
                    st.markdown("---")
                    st.subheader("📊 Confidence Distribution")
                    fig = plot_predictions_bar(probs, "Model Confidence")
                    st.pyplot(fig, use_container_width=True)
        else:
            st.info("👆 Upload an image to see predictions")


# ========================================
# TAB 2: FEATURE EXTRACTION
# ========================================
with tab2:
    st.markdown('<h2 class="section-title">Feature Extraction Models</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload Image")
        uploaded_file2 = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'], key='tab2_upload')
        
        if uploaded_file2 is not None:
            image = Image.open(uploaded_file2).convert('RGB')
            st.image(image, caption='Uploaded Image', width="content")
        else:
            st.info("📌 Upload a plant disease image")
    
    with col2:
        st.subheader("🔍 Model Information")
        
        if model_type in ["ResNet50", "MobileNetV2"]:
            if uploaded_file2 is not None:
                with st.spinner('🔄 Extracting features...'):
                    img_tensor = preprocess_image(image, cfg['image_size'])
                    
                    if model_type == "ResNet50":
                        model = load_resnet50_model(device)
                        model_info = "ResNet50 - Pre-trained on ImageNet (2048 features)"
                    else:
                        model = load_mobilenetv2_model(device)
                        model_info = "MobileNetV2 - Pre-trained on ImageNet (1280 features)"
                    
                    if model is not None:
                        with torch.no_grad():
                            features = model(img_tensor.to(device))
                        
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.success(f"✅ Extracted {features.shape[1]} dimensional feature vector")
                        st.write(f"**Feature Shape:** {tuple(features.shape)}")
                        st.write(f"**Model:** {model_info}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Feature statistics
                        feat_np = features[0].cpu().numpy()
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Mean", f"{feat_np.mean():.4f}")
                        with col_b:
                            st.metric("Std Dev", f"{feat_np.std():.4f}")
                        with col_c:
                            st.metric("Max Value", f"{feat_np.max():.4f}")
                    else:
                        st.error("Model not loaded")
            else:
                st.info("👆 Upload an image to extract features")
        
        elif model_type in ["LSTM", "GRU"]:
            st.write(f"**Model:** {model_type} Sequence Model")
            st.write("Processes sequence of extracted features")
            st.write("Input: 2048-dim ResNet50 features")
            st.write(f"Output: {len(CLASS_NAMES)} class probabilities")
            
            if uploaded_file2 is not None:
                with st.spinner('🔄 Processing sequence...'):
                    img_tensor = preprocess_image(image, cfg['image_size'])
                    resnet = load_resnet50_model(device)
                    
                    if model_type == "LSTM":
                        seq_model = load_lstm_model(device)
                    else:
                        seq_model = load_gru_model(device)
                    
                    if seq_model is None:
                        st.error("Model not trained yet. Run:")
                        st.code("python train.py --review 2")
                    else:
                        with torch.no_grad():
                            features = resnet(img_tensor.to(device)).unsqueeze(1)
                            # Replicate for sequence length
                            features = features.repeat(1, 5, 1)
                            output = seq_model(features)
                            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
                        
                        pred_idx = np.argmax(probs)
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.success(f"🌿 {get_class_display_name(CLASS_NAMES[pred_idx])}")
                        st.metric("Confidence", f"{probs[pred_idx]*100:.2f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("👆 Upload an image for prediction")
        else:
            st.info("Select ResNet50, MobileNetV2, LSTM, or GRU")


# ========================================
# ========================================
# TAB 3: GENERATIVE MODELS
# ========================================
with tab3:
    st.markdown('<h2 class="section-title">Generative Models</h2>', unsafe_allow_html=True)
    
    if model_type == "Autoencoder":
        st.subheader("🔄 Autoencoder Reconstruction")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Input Image**")
            uploaded_file3 = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'], key='tab3_upload')
            
            if uploaded_file3 is not None:
                image = Image.open(uploaded_file3).convert('RGB')
                st.image(image, caption='Original', width="content")
        
        with col2:
            st.write("**Reconstructed Image**")
            
            if uploaded_file3 is not None:
                with st.spinner('🔄 Reconstructing...'):
                    img_tensor = preprocess_image(image, cfg['image_size'])
                    ae_model = load_autoencoder_model(device)
                    
                    if ae_model is None:
                        st.error("Autoencoder model not found. Train:")
                        st.code("python train.py --review 3")
                    else:
                        with torch.no_grad():
                            recon, _ = ae_model(img_tensor.to(device))
                            recon_img = recon[0].cpu()
                            recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min())
                            recon_img = recon_img.permute(1, 2, 0).numpy()
                        
                        st.image(recon_img, caption='Reconstructed', width="content")
            else:
                st.info("👆 Upload an image")
    
    elif model_type == "GAN":
        st.subheader("🎨 GAN Generated Images")
        
        if st.button("Generate New Images", key="gen_button"):
            with st.spinner('🎨 Generating...'):
                gen_model = load_gan_generator(device)
                
                if gen_model is None:
                    st.error("GAN model not found. Train:")
                    st.code("python train.py --review 3")
                else:
                    with torch.no_grad():
                        z = torch.randn(4, 100, 1, 1, device=device)
                        fake_images = gen_model(z)
                        fake_images = (fake_images - fake_images.min()) / (fake_images.max() - fake_images.min())
                    
                    cols = st.columns(4)
                    for i in range(4):
                        with cols[i]:
                            img = fake_images[i].cpu().permute(1, 2, 0).numpy()
                            st.image(img, caption=f'Generated {i+1}', width="content")
        else:
            st.info("Click 'Generate New Images' to create synthetic plant disease images")
    
    else:
        st.info("Select Autoencoder or GAN for generative models")


# ========================================
# TAB 4: ABOUT
# ========================================
with tab4:
    st.markdown('<h2 class="section-title">About This System</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🌱 DL-Plant-Disease-System
    
    A comprehensive Deep Learning system for plant disease classification with 4 academic reviews:
    
    ### Review 1: Classification Baselines
    - **CNN Classifier**: Convolutional Neural Network with 3 conv blocks
    - **MLP Classifier**: Multi-Layer Perceptron baseline
    
    ### Review 2: Transfer Learning + Temporal
    - **ResNet50**: Pre-trained feature extractor (ImageNet)
    - **MobileNetV2**: Efficient pre-trained extractor
    - **LSTM/GRU/RNN**: Sequence models on CNN features
    
    ### Review 3: Generative Models
    - **Autoencoder**: Unsupervised feature learning & reconstruction
    - **DCGAN**: Generative Adversarial Network for synthesis
    
    ### Review 4: End-to-End System
    - **CNN Ensemble**: Final production model
    
    ### Dataset
    - **15 Plant Disease Classes**:
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - Pepper Bacterial spot
        - Pepper Healthy
        - Potato Early blight
        - Potato Late blight
        - Potato Healthy
        - Tomato Bacterial spot
        - Tomato Early blight
        - Tomato Late blight
        - Tomato Leaf Mold
        """)
    
    with col2:
        st.markdown("""
        - Tomato Septoria leaf spot
        - Tomato Spider mites
        - Tomato Target Spot
        - Tomato Mosaic virus
        - Tomato Yellow Leaf Curl Virus
        - Tomato Healthy
        """)
    
    st.markdown("---")
    st.markdown("""
    ### Training
    Train individual reviews:
    ```bash
    python train.py --review 1  # CNN/MLP
    python train.py --review 2  # Transfer Learning
    python train.py --review 3  # Generative Models
    python train.py --review 4  # End-to-End
    ```
    
    ### Configuration
    - **Image Size**: 128×128 pixels
    - **Batch Size**: 32
    - **Epochs**: 15
    - **Optimizer**: Adam (lr=0.0001)
    - **Loss**: CrossEntropyLoss
    """)


# ========================================
# FOOTER
# ========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
<p><strong>DL-Plant-Disease-System</strong> | 15 Plant Disease Classes | Production-Ready Deep Learning</p>
<p>Models saved in: <code>outputs/models/</code> | Results in: <code>outputs/results/</code></p>
</div>
""", unsafe_allow_html=True)
