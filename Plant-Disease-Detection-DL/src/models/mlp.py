"""
MLP (Multi-Layer Perceptron) Model for Plant Disease Detection
Architecture: Flatten -> Dense layers with dropout
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def create_mlp_model(input_shape, num_classes, dropout_rate=0.5):
    """
    Create MLP model
    
    Args:
        input_shape: Input shape of images (height, width, channels)
        num_classes: Number of disease classes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(512, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(256, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_mlp_model(model, learning_rate=0.001):
    """Compile MLP model with optimizer and loss"""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
    )
    return model


if __name__ == "__main__":
    # Example usage
    model = create_mlp_model(input_shape=(224, 224, 3), num_classes=10)
    model = compile_mlp_model(model)
    model.summary()
