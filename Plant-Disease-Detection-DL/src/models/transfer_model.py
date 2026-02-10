"""
Transfer Learning Model using MobileNetV2 for Plant Disease Detection
Pre-trained on ImageNet with fine-tuning capability
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def create_transfer_learning_model(input_shape, num_classes, freeze_base=True, 
                                   unfreeze_layers=30):
    """
    Create Transfer Learning model using MobileNetV2
    
    Args:
        input_shape: Input shape of images (height, width, channels)
        num_classes: Number of disease classes
        freeze_base: Whether to freeze base model initially
        unfreeze_layers: Number of layers to unfreeze from the top
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    if freeze_base:
        base_model.trainable = False
    
    # Add custom top layers
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1.0 / 127.5, offset=-1),  # Normalize for MobileNetV2
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model


def compile_transfer_model(model, learning_rate=0.001):
    """Compile transfer learning model"""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
    )
    return model


def unfreeze_base_model(model, num_layers_to_unfreeze=30):
    """Unfreeze base model layers for fine-tuning"""
    # Find base model index in the Sequential model
    base_model = model.layers[2]  # MobileNetV2 is the 3rd layer (0-indexed: input, rescaling, base_model)
    
    # Unfreeze last N layers
    for layer in base_model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True
    
    return model


def compile_for_finetuning(model, learning_rate=0.0001):
    """Compile model with lower learning rate for fine-tuning"""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
    )
    return model


if __name__ == "__main__":
    # Example usage
    model, base_model = create_transfer_learning_model(
        input_shape=(224, 224, 3),
        num_classes=10,
        freeze_base=True
    )
    model = compile_transfer_model(model)
    model.summary()
