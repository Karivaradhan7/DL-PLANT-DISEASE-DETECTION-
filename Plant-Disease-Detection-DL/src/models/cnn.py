"""
CNN (Convolutional Neural Network) Model for Plant Disease Detection
Architecture: Conv layers -> MaxPooling -> Dense layers
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def create_cnn_model(input_shape, num_classes, dropout_rate=0.5):
    """
    Create CNN model
    
    Args:
        input_shape: Input shape of images (height, width, channels)
        num_classes: Number of disease classes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate * 0.5),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate * 0.5),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate * 0.5),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate * 0.5),
        
        # Global Average Pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(512, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(256, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_cnn_model(model, learning_rate=0.001):
    """Compile CNN model with optimizer and loss"""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
    )
    return model


if __name__ == "__main__":
    # Example usage
    model = create_cnn_model(input_shape=(224, 224, 3), num_classes=10)
    model = compile_cnn_model(model)
    model.summary()
