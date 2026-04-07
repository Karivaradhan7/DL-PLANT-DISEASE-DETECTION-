# DL-Plant-Disease-System

## Problem Statement

Plant diseases threaten food security. This repository provides a complete deep learning pipeline for plant disease classification, generative modeling, temporal analysis, and deployable inference.

## Dataset

Using publicly available plant disease image datasets (e.g., PlantVillage). Images organized in `data/plant_disease/<class_name>/*.jpg`.

## Review 1 - MLP vs CNN Classification

- MLP and CNN models in `src/models/classifiers.py`.
- Trained on dataset with `src/data/dataloader.py`.
- Outputs: accuracy/loss plots, confusion matrix, comparison table in `outputs/results/r1_*`.

## Review 2 - Pretrained CNN + RNN/LSTM/GRU + Attention

- Feature extraction from ResNet50 and MobileNetV2.
- Sequence generation in `src/data/sequence_dataset.py`.
- Models in `src/models/temporal.py`.
- Train and evaluate all variants; include attention.

## Review 3 - Autoencoder + GAN

- Convolutional Autoencoder in `src/models/autoencoder.py`.
- DCGAN in `src/models/dcgan.py` with BatchNorm and label smoothing.
- Outputs: reconstructed images, GAN samples, loss curves, latent visualization via PCA/t-SNE.

## Review 4 - End-to-End System

- Data pipeline with augmentation: `src/data/dataloader.py` and `src/utils/augmentations.py`.
- Training and evaluation pipeline: `train.py`, `src/utils/trainer.py`.
- Metrics: accuracy, precision, recall, F1.
- Streamlit app: `app/app.py`.
- Reproducibility via `config.yaml` and fixed `seed`.

## Running

1. `cd DL-Plant-Disease-System`
2. `python -m pip install -r requirements.txt`
3. `python train.py --review 1` (or 2, 3, 4)
4. `streamlit run app/app.py`

## Academic Evaluation Summary

This project demonstrates academic-grade plant disease detection using deep learning and a Streamlit interface. It includes:

- Project overview: plant disease detection using CNNs, transfer learning, and hybrid models.
- How to run: `cd DL-Plant-Disease-System && python -m pip install -r requirements.txt && streamlit run app/app.py`
- Models used: CNN, MLP, Transfer Learning (MobileNetV2/ResNet), LSTM, GRU, Autoencoder, GAN.
- Dataset: PlantVillage leaf images for healthy and diseased plant classes.

## Results

- Metrics and plots saved in `outputs/results/`.

## Screenshots

- `screenshots/review1.png`
- `screenshots/review2.png`
- `screenshots/review3.png`
- `screenshots/review4.png`
