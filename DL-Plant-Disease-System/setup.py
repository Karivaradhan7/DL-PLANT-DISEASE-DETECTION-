#!/usr/bin/env python3
"""
Quick setup script to generate synthetic data and train models
Run this before launching the app!
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def create_synthetic_dataset(max_samples_per_class=30):
    """Generate synthetic plant disease dataset"""
    dataset_dir = Path("data/plant_disease")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    num_classes = 5
    image_size = 128
    
    print("\n📊 Creating synthetic dataset...")
    
    for class_id in range(num_classes):
        class_dir = dataset_dir / f"class_{class_id}"
        class_dir.mkdir(exist_ok=True)
        
        for img_id in range(max_samples_per_class):
            # Generate random image
            img_array = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
            
            # Add some patterns to make it less uniform
            x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))
            pattern = (np.sin(x/20) * np.sin(y/20) * 50 + 128).astype(np.uint8)
            img_array[..., 0] = np.clip(img_array[..., 0] + pattern, 0, 255)
            
            img = Image.fromarray(img_array)
            img.save(class_dir / f"image_{img_id:03d}.jpg")
        
        print(f"  ✅ Class {class_id}: {max_samples_per_class} images")
    
    print(f"✅ Dataset created at: {dataset_dir}\n")


def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing dependencies...")
    os.system("pip install -q -r requirements.txt")
    print("✅ Dependencies installed\n")


def train_models():
    """Train all models"""
    print("\n🚀 Training models...\n")
    
    reviews = [1, 2, 3, 4]
    for review in reviews:
        print(f"\n{'='*60}")
        print(f"TRAINING REVIEW {review}")
        print(f"{'='*60}\n")
        os.system(f"python train.py --review {review}")


def verify_models():
    """Check if models were saved correctly"""
    print("\n🔍 Verifying model files...\n")
    
    model_paths = [
        "outputs/models/review1/cnn.pth",
        "outputs/models/review1/mlp.pth",
        "outputs/models/review4/cnn.pth",
    ]
    
    all_exist = True
    for path in model_paths:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✅ {path} ({size_mb:.2f} MB)")
        else:
            print(f"  ❌ {path} NOT FOUND")
            all_exist = False
    
    if all_exist:
        print("\n✅ All models found!\n")
        return True
    else:
        print("\n❌ Some models missing. Check training output.\n")
        return False


def main():
    print("\n" + "="*60)
    print("🌱 DL-PLANT-DISEASE-SYSTEM SETUP")
    print("="*60)
    
    # Create synthetic data
    create_synthetic_dataset()
    
    # Install dependencies
    install_dependencies()
    
    # Train models
    train_models()
    
    # Verify
    if verify_models():
        print("\n" + "="*60)
        print("✅ SETUP COMPLETE!")
        print("="*60)
        print("\n🚀 Now you can run the app:\n")
        print("  streamlit run app/app.py\n")
    else:
        print("\n⚠️  Setup incomplete. Check errors above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
