"""
Setup and Launch Plant Disease Detection App
Handles dataset preparation, model training, and app launching
"""
import os
import sys
import subprocess
from pathlib import Path
import json
import tensorflow as tf
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_data
from src.models.cnn import create_cnn_model


def setup_and_train():
    """Setup dataset and train model if not exists"""
    
    saved_models_dir = Path("saved_models")
    saved_models_dir.mkdir(exist_ok=True)
    
    # Check if model already exists
    existing_models = list(saved_models_dir.glob("*.h5")) + list(saved_models_dir.glob("*.keras"))
    if existing_models and Path("saved_models/class_names.json").exists():
        print(f"\n✓ Model already exists: {existing_models[0]}")
        return True
    
    print("\n" + "=" * 60)
    print("Setting up Plant Disease Detection System")
    print("=" * 60)
    
    # Setup dataset
    data_dir = Path("data/raw")
    if not data_dir.exists() or len(list(data_dir.glob("*/"))) == 0:
        print("\n1. Creating dataset...")
        subprocess.run([sys.executable, "download_dataset.py"], check=True)
    else:
        print("\n✓ Dataset already exists")
    
    # Train model
    print("\n2. Training CNN model...")
    try:
        img_size = (128, 128)
        batch_size = 32
        epochs = 1  # Use just 1 epoch for faster startup
        
        print("   Loading dataset...")
        train_ds, val_ds, class_names = load_data(
            data_dir=str(data_dir),
            img_size=img_size,
            batch_size=batch_size
        )
        
        num_classes = len(class_names)
        print(f"   Classes: {num_classes}")
        print(f"   Creating model...")
        
        input_shape = (128, 128, 3)
        model = create_cnn_model(input_shape, num_classes)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        print(f"   Training for {epochs} epochs...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=1
        )
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = saved_models_dir / f"cnn_model_{timestamp}.h5"
        model.save(str(model_path))
        
        # Save class names
        class_names_path = saved_models_dir / "class_names.json"
        with open(class_names_path, 'w') as f:
            json.dump(class_names, f)
        
        print(f"\n✓ Model trained and saved: {model_path}")
        print(f"✓ Class names saved: {class_names_path}")
        return True
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False


def launch_app():
    """Launch Streamlit app"""
    print("\n" + "=" * 60)
    print("Launching Plant Disease Detection App")
    print("=" * 60)
    print("\n✓ Starting Streamlit app...")
    print("🌍 Open your browser and navigate to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the app\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(Path("app/app.py")),
            "--logger.level=warning"
        ])
    except KeyboardInterrupt:
        print("\n\nApp stopped.")


if __name__ == "__main__":
    print("\n🌿 Plant Disease Detection - Full App Setup")
    
    # Setup and train
    if setup_and_train():
        # Launch app
        launch_app()
    else:
        print("\n✗ Setup failed. Please check the errors above.")
        sys.exit(1)
