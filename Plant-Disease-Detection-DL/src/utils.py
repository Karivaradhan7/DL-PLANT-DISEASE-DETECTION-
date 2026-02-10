"""
Utility functions for Plant Disease Detection project
Includes preprocessing, visualization, and helper functions
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_training_history(history, figsize=(15, 5)):
    """Plot training history curves"""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    # Learning Rate (if available)
    if 'lr' in history.history:
        axes[2].plot(history.history['lr'])
        axes[2].set_title('Learning Rate')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_yscale('log')
        axes[2].grid(True)
    
    plt.tight_layout()
    return fig


def visualize_sample_batch(dataset, class_names, num_samples=16, figsize=(12, 10)):
    """Visualize a batch of images"""
    images, labels = next(iter(dataset))
    
    num_samples = min(num_samples, len(images))
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx in range(num_samples):
        image = images[idx].numpy()
        label = labels[idx].numpy()
        
        # If image is normalized, clip to [0, 1]
        if image.max() <= 1.0:
            image = np.clip(image, 0, 1)
        
        axes[idx].imshow(image)
        axes[idx].set_title(class_names[label])
        axes[idx].axis('off')
    
    # Hide remaining axes
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return seed


def get_dataset_statistics(dataset, num_batches=None):
    """Get statistics about dataset"""
    num_samples = 0
    num_batches_processed = 0
    
    for batch_images, batch_labels in dataset:
        num_samples += len(batch_images)
        num_batches_processed += 1
        
        if num_batches is not None and num_batches_processed >= num_batches:
            break
    
    return {
        'num_samples': num_samples,
        'num_batches': num_batches_processed
    }


def create_directories(base_dir, directories):
    """Create multiple directories"""
    base_path = Path(base_dir)
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return base_path


if __name__ == "__main__":
    print("Utility functions loaded successfully.")
