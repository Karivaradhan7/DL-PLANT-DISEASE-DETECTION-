#!/usr/bin/env python3
"""
Test script to verify the entire training and inference pipeline
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

def test_model_training():
    """Test training pipeline"""
    print("\n" + "="*60)
    print("TEST 1: Model Training")
    print("="*60 + "\n")
    
    try:
        print("Running: python train.py --review 1")
        result = os.system("python train.py --review 1 > /tmp/train_test.log 2>&1")
        
        if result == 0:
            print("✅ Training completed successfully\n")
            return True
        else:
            print("❌ Training failed\n")
            return False
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return False


def test_model_loading():
    """Test model loading"""
    print("\n" + "="*60)
    print("TEST 2: Model Loading")
    print("="*60 + "\n")
    
    try:
        from src.models.classifiers import CNNClassifier
        from src.utils.misc import get_device
        
        device = get_device(use_gpu=False)
        model = CNNClassifier(num_classes=5)
        
        model_paths = [
            "outputs/models/review1/cnn.pth",
            "outputs/results/review1/cnn_model.pt",
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                print(f"Loading model from: {model_path}")
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                model.eval()
                print(f"✅ Model loaded successfully\n")
                return True
        
        print(f"❌ No model file found\n")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}\n")
        return False


def test_prediction():
    """Test end-to-end prediction"""
    print("\n" + "="*60)
    print("TEST 3: End-to-End Prediction")
    print("="*60 + "\n")
    
    try:
        from src.models.classifiers import CNNClassifier
        from src.utils.misc import get_device
        
        device = get_device(use_gpu=False)
        model = CNNClassifier(num_classes=5)
        
        # Try to load model
        model_paths = [
            "outputs/models/review1/cnn.pth",
            "outputs/results/review1/cnn_model.pt",
        ]
        
        model_loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                model.eval()
                model_loaded = True
                break
        
        if not model_loaded:
            print("❌ Model not found\n")
            return False
        
        # Create dummy image
        print("Creating synthetic image...")
        img_array = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img).unsqueeze(0)
        
        # Predict
        print("Running inference...")
        with torch.no_grad():
            output = model(img_tensor.to(device))
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        
        print(f"✅ Prediction successful!")
        print(f"   - Predicted class: {pred_class}")
        print(f"   - Confidence: {confidence*100:.2f}%\n")
        return True
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_streamlit_launch():
    """Test streamlit app launch"""
    print("\n" + "="*60)
    print("TEST 4: Streamlit App Availability")
    print("="*60 + "\n")
    
    try:
        result = os.system("streamlit --version > /dev/null 2>&1")
        if result == 0:
            print("✅ Streamlit is installed\n")
            print("To launch the app, run:")
            print("  streamlit run app/app.py\n")
            return True
        else:
            print("❌ Streamlit not found\n")
            return False
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return False


def main():
    print("\n" + "="*60)
    print("🌱 DL-PLANT-DISEASE-SYSTEM PIPELINE TEST")
    print("="*60)
    
    results = {
        "Training": test_model_training(),
        "Loading": test_model_loading(),
        "Prediction": test_prediction(),
        "Streamlit": test_streamlit_launch(),
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60 + "\n")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:20} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("="*60)
        print("\n🚀 Your app is ready! Run:")
        print("   streamlit run app/app.py\n")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease check the errors above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
