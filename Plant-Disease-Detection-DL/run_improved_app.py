"""
Updated Setup and Launch Script using Improved Transfer Learning Model
Automatically trains improved EfficientNetB0 model and launches the app
"""
import os
import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))


def setup_and_train():
    """Setup dataset and train improved model if not exists"""
    
    saved_models_dir = Path("saved_models")
    saved_models_dir.mkdir(exist_ok=True)
    
    # Check if good model already exists
    h5_models = list(saved_models_dir.glob("efficientnet_model*.h5"))
    savedmodel_dir = list(saved_models_dir.glob("efficientnet_savedmodel*"))
    
    if (h5_models or savedmodel_dir) and Path("saved_models/class_names.json").exists():
        print(f"\n✓ Trained model already exists!")
        return True
    
    print("\n" + "=" * 70)
    print("🌿 SETTING UP PLANT DISEASE DETECTION SYSTEM")
    print("=" * 70)
    
    # Setup dataset
    data_dir = Path("data/raw")
    if not data_dir.exists() or len(list(data_dir.glob("*/"))) == 0:
        print("\n1. Creating dataset...")
        try:
            subprocess.run([sys.executable, "download_dataset.py"], check=True)
        except Exception as e:
            print(f"✗ Error downloading dataset: {e}")
            return False
    else:
        print("\n✓ Dataset already exists")
    
    # Train improved model
    print("\n2. Training improved EfficientNetB0 model...")
    try:
        subprocess.run([sys.executable, "train_improved.py"], check=True)
        return True
    except Exception as e:
        print(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False


def launch_app():
    """Launch Streamlit app"""
    print("\n" + "=" * 70)
    print("🚀 LAUNCHING PLANT DISEASE DETECTION APP")
    print("=" * 70)
    print("\n✓ Starting Streamlit app...")
    print("🌍 Open your browser and navigate to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the app\n")
    print("=" * 70 + "\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(Path("app/app.py")),
            "--logger.level=warning"
        ])
    except KeyboardInterrupt:
        print("\n\n✓ App stopped gracefully.")


if __name__ == "__main__":
    print("\n🌿 PLANT DISEASE DETECTION - IMPROVED TRAINING & APP LAUNCHER")
    
    # Setup and train
    if setup_and_train():
        # Launch app
        launch_app()
    else:
        print("\n✗ Setup failed. Please check the errors above.")
        sys.exit(1)
