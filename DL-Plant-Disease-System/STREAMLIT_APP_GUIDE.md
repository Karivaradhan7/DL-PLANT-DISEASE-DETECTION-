# 🌱 DL-Plant-Disease-System Streamlit App Guide

## 📋 Overview

Production-quality Streamlit application featuring 4 main tabs integrating all review components:
- **Tab 1**: CNN/MLP Classification with Grad-CAM
- **Tab 2**: Sequential + Transfer Learning Models
- **Tab 3**: Time Series & Temporal Analysis
- **Tab 4**: Generative Models (Autoencoder + GAN)

---

## 🚀 Quick Start

```bash
# Navigate to project root
cd DL-Plant-Disease-System

# Install dependencies (if not already done)
pip install -r requirements.txt

# Train models first (required)
python train.py --review 1
python train.py --review 2
python train.py --review 3
python train.py --review 4

# Launch Streamlit app
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🎛️ Sidebar Features

### Model Selection
- **CNN**: Convolutional Neural Network (Review 1)
- **MLP**: Multi-Layer Perceptron (Review 1)
- **LSTM**: Long Short-Term Memory (Review 2)
- **GRU**: Gated Recurrent Unit (Review 2)
- **ResNet50**: Pretrained transfer learning (Review 2)
- **MobileNetV2**: Lightweight transfer learning (Review 2)
- **Autoencoder**: Generative model (Review 3)
- **GAN**: Generative Adversarial Network (Review 3)

### Device Selection
- **CPU**: Standard CPU inference
- **GPU**: CUDA GPU acceleration (if available)

---

## 📑 Tab Features & Usage

### Tab 1: CNN/MLP Classification (Review 1)

#### Image Upload & Prediction
1. Upload a plant image (JPG/PNG)
2. Model automatically processes the image
3. Get instant prediction with confidence score

#### Outputs
- **Predicted Class**: Top prediction with confidence percentage
- **Confidence Distribution**: Bar chart of class probabilities
- **Grad-CAM Visualization**: Attention heatmap showing model focus areas (CNN only)
- **Saved Results**: Lists all training outputs

#### Expected Files
```
outputs/results/review1/
├── cnn_loss.png           # Training loss curve
├── cnn_acc.png            # Accuracy curve
├── cnn_confusion.png      # Confusion matrix
├── cnn_model.pt           # Trained model
├── mlp_loss.png
├── mlp_acc.png
├── mlp_confusion.png
└── mlp_model.pt
```

---

### Tab 2: Sequential + Transfer Learning (Review 2)

#### Model Selection
- **RNN Type**: Choose between LSTM, GRU, or RNN
- **Attention**: Enable/disable attention mechanism
- **Pretrained CNN**: ResNet50 or MobileNetV2 for feature extraction

#### Comparison Tables
- **Model Comparison**: Parameters, convergence speed
- **Transfer Learning**: Model sizes and baseline accuracy

#### Feature Extraction
- Upload image → Extract features via ResNet50
- Display feature vector dimensions (2048D)
- Load LSTM model for sequence prediction

#### Expected Files
```
outputs/results/review2/
├── resnet50_LSTM_attn_False.pt
├── resnet50_LSTM_attn_True.pt
├── resnet50_GRU_attn_False.pt
├── resnet50_GRU_attn_True.pt
├── resnet50_RNN_attn_False.pt
└── ... (MobileNetV2 variants)
```

---

### Tab 3: Time Series Models (Review 2 Temporal)

#### Temporal Data Visualization
- **Sequence Configuration**: Adjust sequence length and sample count
- **Synthetic Time Series**: Generated plots of disease progression
- **Attention Weights**: Self-attention heatmap visualization

#### Visualizations
- 3 interactive sequence plots showing temporal patterns
- Self-attention weights heatmap (sequence × sequence)
- Single position attention distribution

#### Model Parameters Display
- Hidden size, input features, output classes
- Real-time configuration from `config.yaml`

---

### Tab 4: Generative Models (Review 3)

#### Sub-Tab 4.1: Autoencoder 🔄

**Configuration Display**
- Latent dimension: 64
- Input/output size: 128×128×3
- Encoder channels: [32, 64, 128]

**Image Reconstruction**
1. Upload image for reconstruction
2. Shows:
   - Original image
   - Reconstructed image
   - Latent statistics (mean, std)

**Architecture**
- Encoder: 3 Conv layers → Latent space
- Decoder: 3 DeConv layers → Reconstruction

#### Sub-Tab 4.2: GAN 🎲

**DCGAN Configuration**
- Latent dimension: 100
- Generator channels: [1024, 512, 256, 128, 64]
- Stability tricks: BatchNorm, Label smoothing

**Generate Synthetic Images**
1. Slider to select number of samples (1-16)
2. Random noise → Generator → Synthetic images
3. Display grid of generated samples

**Stability Features**
✅ Batch Normalization  
✅ Label smoothing (0.9/0.1)  
✅ LeakyReLU activation  
✅ Spectral Norm Ready  

#### Sub-Tab 4.3: Latent Space 📊

**Dimensionality Reduction Visualization**
- **PCA**: Linear dimensionality reduction (fast)
- **t-SNE**: Nonlinear manifold learning (slower, better separation)

**Interactive Plot**
- 2D scatter plot of classes
- Color-coded by class label
- Shows class clustering and separation

**Statistics Display**
- Total points in latent space
- Number of dimensions
- Number of classes
- Points per class

#### Expected Files
```
outputs/results/review3/
├── autoencoder.pt              # Trained AE weights
├── gan_generator.pt            # Generator weights
├── gan_discriminator.pt        # Discriminator weights
├── ae_input.png               # Sample input images
├── ae_recon.png               # Reconstructed images
├── gan_samples.png            # Generated samples
├── latent_pca.png             # PCA visualization
└── latent_tsne.png            # t-SNE visualization
```

---

## 🎨 UI Components & Features

### Caching & Performance
- **Model Loading**: `@st.cache_resource` for instant model reuse
- **Config Loading**: Cached YAML configuration
- **Optimized Preprocessing**: Efficient image transforms

### Error Handling
- Graceful fallbacks if model files missing
- Clear user messages for missing dependencies
- Device availability checks

### Visual Design
- **Clean Layout**: st.columns for responsive design
- **Color Scheme**: Green (#2ecc71), Blue (#3498db), Red (#e74c3c)
- **Section Styling**: Custom HTML/CSS for visual hierarchy
- **Icons & Emojis**: Professional UI with visual indicators

### User Feedback
- Loading spinners during processing
- Success/error message boxes
- Metric displays with confidence scores
- Expandable sections for detailed information

---

## 🔧 Advanced Features

### Grad-CAM Visualization

**How It Works**
1. Hooks into last convolutional layer
2. Computes feature importance via gradients
3. Generates attention heatmap
4. Overlays on activation map

**Supported Models**
- ✅ CNN/CNNClassifier
- ❌ MLP (not applicable - no spatial structure)

**Interpretation**
- Red areas: High model attention
- Blue areas: Low attention
- Shows which image regions drove the prediction

### Model Comparison

**Tab 2 Summary**
- LSTM: Most parameters (~1.2M), balanced convergence
- GRU: Fewer parameters (~0.9M), faster convergence
- RNN: Fewest parameters (~0.8M), slow convergence

**Transfer Learning Summary**
- ResNet50: Larger (97.7 MB), better accuracy (76.1%)
- MobileNetV2: Smaller (13.5 MB), good accuracy (71.3%)

---

## 📊 Configuration Management

### Loading from `config.yaml`

The app automatically reads:
```yaml
seed: 42
batch_size: 32
num_epochs: 10
image_size: 128
classes: 5
experiments:
  review1:
    mlp_hidden: 256
    cnn_channels: [32, 64, 128]
  review2:
    sequence_len: 5
    hidden_size: 128
  review3:
    ae_latent_dim: 64
    gan_latent_dim: 100
```

---

## 🛠️ Troubleshooting

### Issue: "Model weights not found"
**Solution**: Run training first
```bash
python train.py --review 1  # or 2, 3, 4
```

### Issue: GPU not being used
**Solution**: 
1. Check CUDA availability: `torch.cuda.is_available()`
2. Select GPU from sidebar
3. Ensure PyTorch CUDA version matches GPU driver

### Issue: "No module named src.models"
**Solution**: Ensure running from project root
```bash
cd DL-Plant-Disease-System
streamlit run app/app.py
```

### Issue: Streamlit cache issues
**Solution**: Clear cache
```bash
streamlit cache clear
streamlit run app/app.py
```

---

## 📈 Expected Performance

### Tab 1: CNN/MLP Inference
- Time: ~100-500ms per prediction
- GPU: ~50-100ms
- Accuracy: 85-95% (depends on dataset)

### Tab 2: Feature Extraction
- ResNet50: ~200-300ms (GPU)
- LSTM prediction: ~50ms

### Tab 3: Time Series
- Synthetic data generation: ~100ms
- Visualization: ~200ms

### Tab 4: Generative Models
- Autoencoder reconstruction: ~150ms
- GAN sample generation: ~200ms per sample
- Latent space visualization: ~1-5s (t-SNE slower)

---

## 📚 Dataset Requirements

For full functionality:
```
data/plant_disease/
├── Class0/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── Class1/
│   ├── ...
├── Class2/
│   └── ...
├── Class3/
└── Class4/
```

- **Format**: JPG, PNG
- **Size**: 128×128 (auto-resized)
- **Classes**: 5 (configurable)
- **Split**: 70% train, 15% val, 15% test

---

## 🎯 Next Steps

1. **Train Models**: `python train.py --review 1-4`
2. **Launch App**: `streamlit run app/app.py`
3. **Test Predictions**: Upload sample images
4. **Fine-tune**: Modify `config.yaml` as needed
5. **Deploy**: Use Streamlit Cloud or Docker

---

## 📝 Code Structure

```
app/
└── app.py (800+ lines)
    ├── Config & Setup (40 lines)
    ├── Helper Functions (200 lines)
    │   ├── Model loaders
    │   ├── Preprocessing
    │   ├── Prediction
    │   ├── Grad-CAM
    │   └── Visualization
    ├── Sidebar Config (50 lines)
    ├── Tab 1: CNN/MLP (200 lines)
    ├── Tab 2: Sequential+Transfer (150 lines)
    ├── Tab 3: Time Series (150 lines)
    ├── Tab 4: Generative (300 lines)
    │   ├── Autoencoder (100 lines)
    │   ├── GAN (100 lines)
    │   └── Latent Space (100 lines)
    └── Footer & Info (50 lines)
```

---

## ✨ Features Implemented

✅ 4 distinct tabs covering all review components  
✅ Sidebar model & device selection  
✅ Real-time image prediction & confidence scores  
✅ Grad-CAM attention visualization  
✅ CNN vs MLP comparison  
✅ LSTM/GRU/RNN with attention support  
✅ ResNet50 & MobileNetV2 feature extraction  
✅ Autoencoder reconstruction visualization  
✅ DCGAN synthetic image generation  
✅ Latent space PCA/t-SNE visualization  
✅ Cached model loading for performance  
✅ Error handling & user-friendly messages  
✅ Professional UI with custom styling  
✅ Config.yaml integration  
✅ GPU/CPU device selection  

---

## 🚀 Ready to Deploy!

Your production-quality Streamlit app is ready:
```bash
streamlit run DL-Plant-Disease-System/app/app.py
```

For Streamlit Cloud deployment, add `requirements.txt` to repo and connect your GitHub account!
