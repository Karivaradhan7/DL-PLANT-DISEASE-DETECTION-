"""
Transfer Learning Model using EfficientNetB0 for Plant Disease Detection
Pre-trained on ImageNet with fine-tuning capability
Optimized for high accuracy and confidence scores
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def create_efficientnet_model(input_shape=(224, 224, 3), num_classes=38, dropout_rate=0.4):
    """
    Create Transfer Learning model using EfficientNetB0
    
    Args:
        input_shape: Input shape of images (height, width, channels)
        num_classes: Number of disease classes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        model: Compiled Keras model ready for initial training (base frozen)
    """
    
    # Load pre-trained EfficientNetB0 from ImageNet
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model for initial training
    base_model.trainable = False
    
    # Build custom top layers
    inputs = layers.Input(shape=input_shape)
    
    # Apply appropriate preprocessing for EfficientNet
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    
    # Pass through base model
    x = base_model(x, training=False)
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Batch normalization
    x = layers.BatchNormalization()(x)
    
    # Dropout
    x = layers.Dropout(dropout_rate)(x)
    
    # Dense layer with batch norm
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate * 0.75)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model, base_model


def compile_model(model, learning_rate=0.001):
    """Compile model for initial training (base frozen)"""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def unfreeze_and_compile_finetuning(model, num_layers_to_unfreeze=50, learning_rate=0.0001):
    """
    Unfreeze base model layers for fine-tuning and recompile
    
    Args:
        model: The trained model
        num_layers_to_unfreeze: Number of base model layers to unfreeze from the top
        learning_rate: Lower learning rate for fine-tuning
        
    Returns:
        model: Unfrozen and recompiled model
    """
    
    # Find and unfreeze base model
    base_model = model.layers[2]  # EfficientNetB0 is typically the 3rd layer
    base_model.trainable = True
    
    # Unfreeze only the last N layers
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


if __name__ == "__main__":
    # Example usage
    model, base_model = create_efficientnet_model(
        input_shape=(224, 224, 3),
        num_classes=38
    )
    model = compile_model(model)
    model.summary()
