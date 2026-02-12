"""
Quick training script for Plant Disease Detection
Trains a simple CNN model on the Plant Village dataset
"""
import os
import sys
import tensorflow as tf
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_data
from src.models.cnn import create_cnn_model


def quick_train():
    """Train a quick CNN model"""
    
    data_dir = "data/raw"
    img_size = (128, 128)
    batch_size = 32
    epochs = 3  # Quick training with just 3 epochs
    
    print("=" * 60)
    print("Plant Disease Detection - Quick Model Training")
    print("=" * 60)
    
    print("\n1. Loading dataset...")
    try:
        train_ds, val_ds, class_names = load_data(
            data_dir=data_dir,
            img_size=img_size,
            batch_size=batch_size
        )
        num_classes = len(class_names)
        print(f"✓ Dataset loaded successfully!")
        print(f"  - Classes: {num_classes}")
        print(f"  - Image size: {img_size}")
        print(f"  - Batch size: {batch_size}")
        print(f"\nClasses: {', '.join(class_names[:5])}...")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False
    
    print("\n2. Creating model...")
    input_shape = (128, 128, 3)
    try:
        model = create_cnn_model(input_shape, num_classes)
        print("✓ CNN model created successfully!")
        model.summary()
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return False
    
    print("\n3. Compiling model...")
    try:
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        print("✓ Model compiled successfully!")
    except Exception as e:
        print(f"✗ Error compiling model: {e}")
        return False
    
    print(f"\n4. Training model for {epochs} epochs...")
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=1
        )
        print("✓ Model trained successfully!")
    except Exception as e:
        print(f"✗ Error training model: {e}")
        return False
    
    print("\n5. Saving model...")
    try:
        save_dir = Path("saved_models")
        save_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = save_dir / f"cnn_{timestamp}.keras"
        
        model.save(model_path)
        print(f"✓ Model saved to: {model_path}")
        
        # Also save class names
        import json
        class_names_path = save_dir / "class_names.json"
        with open(class_names_path, 'w') as f:
            json.dump(class_names, f)
        print(f"✓ Class names saved to: {class_names_path}")
        
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("Training completed successfully! ✓")
    print("=" * 60)
    print(f"\nModel ready for inference:")
    print(f"  Model path: {model_path}")
    print(f"  Classes: {num_classes}")
    
    return True


if __name__ == "__main__":
    success = quick_train()
    sys.exit(0 if success else 1)
