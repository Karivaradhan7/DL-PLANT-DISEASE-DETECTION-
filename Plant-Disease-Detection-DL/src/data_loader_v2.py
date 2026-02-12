"""
Improved Data Loader with Data Augmentation for Plant Disease Detection
Includes proper preprocessing, augmentation, and handling of imbalanced datasets
"""

import tensorflow as tf
from tensorflow.keras import layers, preprocessing
import numpy as np
from collections import Counter


def load_data_with_augmentation(data_dir, img_size=(224, 224), batch_size=32):
    """
    Load PlantVillage dataset with data augmentation
    
    Args:
        data_dir: Path to dataset directory
        img_size: Image size tuple (height, width)
        batch_size: Batch size for training
        
    Returns:
        train_ds: Augmented training dataset
        val_ds: Validation dataset (no augmentation)
        class_names: List of class names
    """
    
    # First load data without batching to get class names
    train_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int'
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int'
    )
    
    class_names = train_ds_raw.class_names
    
    # Create data augmentation pipeline
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ], name="data_augmentation")
    
    # Create preprocessing pipeline
    def preprocess_fn(images, labels):
        """Apply augmentation and preprocessing"""
        # Apply augmentation
        images = data_augmentation(images, training=True)
        return images, labels
    
    # Apply augmentation
    train_ds = train_ds_raw.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    # Optimize validation dataset
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, class_names


def load_data_for_processing(data_dir, img_size=(224, 224), batch_size=32):
    """
    Load PlantVillage dataset without augmentation for analysis
    
    Args:
        data_dir: Path to dataset directory
        img_size: Image size tuple (height, width)
        batch_size: Batch size
        
    Returns:
        train_ds: Training dataset
        val_ds: Validation dataset
        class_names: List of class names
    """
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )
    
    class_names = train_ds.class_names
    
    return train_ds, val_ds, class_names


def calculate_class_weights(data_dir):
    """
    Calculate class weights for imbalanced dataset
    
    Args:
        data_dir: Path to dataset directory
        
    Returns:
        class_weights: Dictionary with class weights
    """
    from pathlib import Path
    
    class_counts = {}
    data_path = Path(data_dir)
    
    # Count samples per class
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            num_samples = len(list(class_dir.glob("*.*")))
            class_name = class_dir.name
            class_counts[class_name] = num_samples
    
    # Calculate weights (inverse frequency)
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    class_weights = {}
    for idx, (class_name, count) in enumerate(sorted(class_counts.items())):
        weight = (total_samples / (num_classes * count)) if count > 0 else 1.0
        class_weights[idx] = weight
    
    return class_weights, class_counts


if __name__ == "__main__":
    # Example usage
    train_ds, val_ds, class_names = load_data_with_augmentation(
        "data/raw",
        img_size=(224, 224),
        batch_size=32
    )
    print(f"Classes: {len(class_names)}")
    print(f"Classes: {class_names}")
