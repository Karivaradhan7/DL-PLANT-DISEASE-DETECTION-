"""
Inference Module for Plant Disease Detection
Load trained models and make predictions on new images
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image


class DiseasePredictor:
    """Load and use trained models for inference"""
    
    def __init__(self, model_path, class_names):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model
            class_names: List of disease class names
        """
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names
        self.img_size = 224
    
    def predict_image(self, image_path):
        """
        Predict disease from image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (predicted_class, confidence, all_predictions)
        """
        image = Image.open(image_path).convert('RGB')
        image_array = self._preprocess_image(image)
        
        predictions = self.model.predict(image_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        return (
            self.class_names[predicted_class_idx],
            float(confidence),
            predictions[0]
        )
    
    def predict_array(self, image_array):
        """
        Predict disease from numpy array
        
        Args:
            image_array: Image as numpy array (H, W, 3)
            
        Returns:
            Tuple of (predicted_class, confidence, all_predictions)
        """
        processed_array = self._preprocess_array(image_array)
        predictions = self.model.predict(processed_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        return (
            self.class_names[predicted_class_idx],
            float(confidence),
            predictions[0]
        )
    
    def batch_predict(self, image_paths):
        """
        Predict diseases for multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            result = self.predict_image(image_path)
            results.append({
                'image': image_path,
                'predicted_class': result[0],
                'confidence': result[1],
                'all_predictions': {
                    self.class_names[i]: float(result[2][i])
                    for i in range(len(self.class_names))
                }
            })
        
        return results
    
    def _preprocess_image(self, image):
        """Preprocess PIL Image"""
        image = image.resize((self.img_size, self.img_size))
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    
    def _preprocess_array(self, image_array):
        """Preprocess numpy array"""
        if image_array.dtype != np.float32:
            image_array = image_array.astype(np.float32) / 255.0
        
        if image_array.shape[0] != self.img_size or image_array.shape[1] != self.img_size:
            pil_image = Image.fromarray((image_array * 255).astype(np.uint8))
            image_array = self._preprocess_image(pil_image)
        else:
            image_array = np.expand_dims(image_array, axis=0)
        
        return image_array


if __name__ == "__main__":
    # Example usage
    model_path = "../../saved_models/best_model.h5"
    class_names = ["Healthy", "Fungal", "Bacterial", "Viral"]
    
    predictor = DiseasePredictor(model_path, class_names)
    
    # Predict single image
    result = predictor.predict_image("sample_image.jpg")
    print(f"Disease: {result[0]}, Confidence: {result[1]:.2f}")
