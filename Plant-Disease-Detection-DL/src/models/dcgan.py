"""
DCGAN (Deep Convolutional Generative Adversarial Network) Model
Used for generating synthetic plant disease images for data augmentation
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def create_generator(latent_dim=100, output_shape=(224, 224, 3)):
    """
    Create Generator model
    
    Args:
        latent_dim: Dimension of latent noise vector
        output_shape: Output image shape (height, width, channels)
        
    Returns:
        Generator model
    """
    model = models.Sequential([
        # Input: latent vector
        layers.Input(shape=(latent_dim,)),
        
        # Dense layer with reshaping
        layers.Dense(28 * 28 * 256, use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Reshape((28, 28, 256)),
        
        # Transpose Convolution layers
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        # Output layer
        layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh')
    ], name='generator')
    
    return model


def create_discriminator(input_shape=(224, 224, 3)):
    """
    Create Discriminator model
    
    Args:
        input_shape: Input image shape (height, width, channels)
        
    Returns:
        Discriminator model
    """
    model = models.Sequential([
        # Input: image
        layers.Input(shape=input_shape),
        
        # Convolution layers
        layers.Conv2D(16, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),
        
        layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),
        
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),
        
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),
        
        # Flattening and output
        layers.Flatten(),
        layers.Dense(1)  # Raw output, will use logits loss
    ], name='discriminator')
    
    return model


def create_gan(generator, discriminator):
    """
    Create combined GAN model for training
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        
    Returns:
        Combined GAN model
    """
    # Freeze discriminator weights during generator training
    discriminator.trainable = False
    
    model = models.Sequential([
        generator,
        discriminator
    ], name='GAN')
    
    return model


def compile_gan_models(generator, discriminator, gan):
    """Compile all GAN models"""
    # Compile discriminator
    discriminator.trainable = True
    discriminator.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Compile generator and GAN
    discriminator.trainable = False
    gan.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    return generator, discriminator, gan


if __name__ == "__main__":
    # Example usage
    generator = create_generator(latent_dim=100, output_shape=(224, 224, 3))
    discriminator = create_discriminator(input_shape=(224, 224, 3))
    gan = create_gan(generator, discriminator)
    
    generator, discriminator, gan = compile_gan_models(generator, discriminator, gan)
    
    generator.summary()
    discriminator.summary()
