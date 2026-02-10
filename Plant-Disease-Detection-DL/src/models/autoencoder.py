"""
Autoencoder Model for Plant Disease Detection
Architecture: Encoder -> Latent Space -> Decoder
Used for anomaly detection and dimensionality reduction
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def create_autoencoder(input_shape, latent_dim=128):
    """
    Create Autoencoder model
    
    Args:
        input_shape: Input shape of images (height, width, channels)
        latent_dim: Dimension of latent space
        
    Returns:
        Tuple of (full_autoencoder, encoder, decoder)
    """
    # Encoder
    encoder_inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255.0)(encoder_inputs)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    x = layers.Flatten()(x)
    encoder_outputs = layers.Dense(latent_dim, activation='relu', name='latent')(x)
    
    encoder = models.Model(encoder_inputs, encoder_outputs, name='encoder')
    
    # Decoder
    decoder_inputs = layers.Input(shape=(latent_dim,))
    
    # Calculate reshape dimensions (224 -> 28 after 3 poolings)
    x = layers.Dense(28 * 28 * 128, activation='relu')(decoder_inputs)
    x = layers.Reshape((28, 28, 128))(x)
    
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    decoder_outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    decoder = models.Model(decoder_inputs, decoder_outputs, name='decoder')
    
    # Full autoencoder
    autoencoder = models.Model(encoder_inputs, decoder(encoder_outputs), name='autoencoder')
    
    return autoencoder, encoder, decoder


def compile_autoencoder(autoencoder, learning_rate=0.001):
    """Compile autoencoder"""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    return autoencoder


if __name__ == "__main__":
    # Example usage
    autoencoder, encoder, decoder = create_autoencoder(
        input_shape=(224, 224, 3),
        latent_dim=128
    )
    autoencoder = compile_autoencoder(autoencoder)
    autoencoder.summary()
