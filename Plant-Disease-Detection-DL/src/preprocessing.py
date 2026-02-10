"""
Image Preprocessing Module for Plant Disease Detection
Handles image normalization, augmentation, and preparation
"""

import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path


class ImagePreprocessor:
    """Preprocess and augment images"""
    
    def __init__(self, target_size=224, normalize=True):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target image size (square)
            normalize: Whether to normalize to [0, 1]
        """
        self.target_size = target_size
        self.normalize = normalize
    
    def load_image(self, image_path):
        """Load image from file"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def resize_image(self, image):
        """Resize image to target size"""
        image = cv2.resize(image, (self.target_size, self.target_size))
        return image
    
    def normalize_image(self, image):
        """Normalize image to [0, 1]"""
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = np.clip(image / 255.0, 0, 1)
        return image
    
    def preprocess(self, image_path):
        """Complete preprocessing pipeline"""
        image = self.load_image(image_path)
        image = self.resize_image(image)
        
        if self.normalize:
            image = self.normalize_image(image)
        
        return image
    
    def batch_preprocess(self, image_paths):
        """Preprocess multiple images"""
        images = []
        
        for image_path in image_paths:
            image = self.preprocess(image_path)
            images.append(image)
        
        return np.array(images)
    
    @staticmethod
    def apply_augmentation(image):
        """Apply random augmentation to image"""
        # Random flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random rotation
        if np.random.rand() > 0.5:
            angle = np.random.randint(-15, 15)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h))
        
        # Random brightness
        if np.random.rand() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, 0, 1 if image.dtype == np.float32 else 255)
        
        return image
    
    @staticmethod
    def apply_histogram_equalization(image):
        """Apply histogram equalization"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert to HSV
            hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(image.dtype) / 255.0
        
        return image
    
    @staticmethod
    def apply_gaussian_blur(image, kernel_size=5):
        """Apply Gaussian blur"""
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return image


if __name__ == "__main__":
    # Example usage
    preprocessor = ImagePreprocessor(target_size=224)
    
    # Load and preprocess single image
    image = preprocessor.preprocess("sample_image.jpg")
    print(f"Preprocessed image shape: {image.shape}")
