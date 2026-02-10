"""
# Plant Disease Detection - Project Structure README

## Overview
A comprehensive deep learning project for plant disease detection using TensorFlow, organized with multiple review stages and final deployment.

## Directory Structure

```
Plant-Disease-Detection-DL/
├── app/                          # Web application for deployment
│   ├── app.py                   # Streamlit web interface
│   └── inference.py             # Model inference module
├── config/                       # Configuration files
│   └── config.yaml              # Project configuration
├── data/                         # Data directory
│   ├── raw/                     # Original dataset
│   ├── processed/               # Preprocessed data
│   └── splits/                  # Train/val/test splits
├── experiments/                  # Experiment tracking
│   ├── R1/                      # Review 1: MLP & CNN
│   ├── R2/                      # Review 2: Transfer Learning
│   └── R3/                      # Review 3: Autoencoder & DCGAN
├── notebooks/                    # Jupyter notebooks for analysis
├── outputs/                      # Project outputs
│   ├── plots/                   # Training/evaluation plots
│   ├── confusion_matrices/      # Confusion matrix visualizations
│   └── generated_images/        # GAN-generated images
├── saved_models/                 # Trained model checkpoints
├── src/                          # Source code
│   ├── models/                  # Model architectures
│   │   ├── mlp.py              # Multi-Layer Perceptron
│   │   ├── cnn.py              # Convolutional Neural Network
│   │   ├── transfer_model.py   # Transfer Learning (MobileNetV2)
│   │   ├── autoencoder.py      # Autoencoder for anomaly detection
│   │   └── dcgan.py            # Deep Convolutional GAN
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── preprocessing.py        # Image preprocessing utilities
│   ├── train.py                # Training pipeline
│   ├── evaluate.py             # Model evaluation metrics
│   ├── inference.py            # Model inference
│   └── utils.py                # Utility functions
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

## Key Components

### Models (src/models/)
- **MLP.py**: Simple Multi-Layer Perceptron for baseline
- **CNN.py**: Custom Convolutional Neural Network with 4 blocks
- **Transfer Learning**: Pre-trained MobileNetV2 with fine-tuning
- **Autoencoder**: Encoder-Decoder architecture for anomaly detection
- **DCGAN**: Generative model for synthetic image generation

### Core Modules (src/)
- **data_loader.py**: Dataset management, augmentation, and splitting
- **preprocessing.py**: Image preprocessing and normalization
- **train.py**: Unified training interface for all models
- **evaluate.py**: Comprehensive evaluation metrics and visualizations
- **utils.py**: Helper functions for plotting and analysis

### Application (app/)
- **app.py**: Streamlit web interface for:
  - Single image prediction
  - Batch processing
  - Model information display
- **inference.py**: Production-ready inference class

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Plant-Disease-Detection-DL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training Models

```python
from src.train import ModelTrainer
from src.data_loader import PlantDiseaseDataLoader

# Load data
loader = PlantDiseaseDataLoader("data/raw", img_size=224)
train_ds, val_ds, test_ds = loader.load_from_directory()

# Train MLP model
trainer = ModelTrainer(model_name='mlp', config={})
trainer.build_model((224, 224, 3), num_classes=10)
trainer.train(train_ds, val_ds, epochs=50)
trainer.save_model()
```

### Evaluation

```python
from src.evaluate import ModelEvaluator

evaluator = ModelEvaluator(model, class_names)
metrics = evaluator.evaluate(test_dataset)
evaluator.plot_confusion_matrix()
evaluator.plot_class_accuracy()
```

### Model Inference

```python
from app.inference import DiseasePredictor

predictor = DiseasePredictor("saved_models/best_model.h5", class_names)
disease, confidence, predictions = predictor.predict_image("leaf.jpg")
print(f"Disease: {disease}, Confidence: {confidence:.2f}")
```

### Web Application

```bash
streamlit run app/app.py
```

Then navigate to `http://localhost:8501` in your browser.

## Project Reviews

### Review 1: MLP and CNN (experiments/R1/)
- Baseline MLP model with 4 dense layers
- Custom CNN with 4 convolutional blocks
- Comparison of fully-connected vs convolutional architectures

### Review 2: Transfer Learning (experiments/R2/)
- MobileNetV2 pre-trained on ImageNet
- Feature extraction with frozen base model
- Fine-tuning with unfrozen layers

### Review 3: Autoencoder and DCGAN (experiments/R3/)
- Autoencoder for anomaly detection
- DCGAN for synthetic image generation
- Data augmentation through generative modeling

### Final Deployment
- Streamlit web application
- REST API (optional)
- Model serving with optimized inference

## Configuration

Edit `config/config.yaml` to customize:
- Data directory paths
- Model hyperparameters
- Training settings
- Class names
- Device settings

## Dependencies

Key libraries:
- **TensorFlow**: Deep learning framework
- **NumPy/Pandas**: Data processing
- **Scikit-learn**: Metrics and utilities
- **OpenCV**: Image processing
- **Matplotlib/Seaborn**: Visualization
- **Streamlit**: Web application
- **Pillow**: Image handling

## Model Performance Tracking

Track experiment results in:
- `experiments/R1/`: First review results
- `experiments/R2/`: Second review results
- `experiments/R3/`: Third review results

Save outputs:
- Models: `saved_models/`
- Plots: `outputs/plots/`
- Confusion matrices: `outputs/confusion_matrices/`
- Generated images: `outputs/generated_images/`

## Best Practices

1. **Data Management**:
   - Keep raw data in `data/raw/`
   - Processed data in `data/processed/`
   - Use data splits from `data/splits/`

2. **Model Development**:
   - Start with MLP/CNN baselines
   - Use transfer learning for better accuracy
   - Implement regularization (dropout, batch norm)

3. **Evaluation**:
   - Always evaluate on separate test set
   - Generate confusion matrices
   - Calculate per-class metrics

4. **Deployment**:
   - Save best models with timestamps
   - Document model architecture
   - Test inference pipeline

## Future Enhancements

- [ ] REST API deployment
- [ ] Model quantization for edge devices
- [ ] Real-time camera feed integration
- [ ] Mobile app development
- [ ] Ensemble modeling
- [ ] Explainability (GradCAM, LIME)

## License

Specify your license here

## Contact

For questions or collaboration, contact: <your-email>
"""

if __name__ == "__main__":
    print(__doc__)
