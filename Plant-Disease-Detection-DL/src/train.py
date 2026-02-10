"""
Simple Training Module for Plant Disease Detection
Supports: mlp, cnn, transfer_learning
"""

import argparse
import tensorflow as tf
from datetime import datetime
from pathlib import Path

from .data_loader import load_data
from .models.mlp import create_mlp_model
from .models.cnn import create_cnn_model
from .models.transfer_model import create_transfer_learning_model


def train_model(model_name):

    data_dir = "data/raw"
    img_size = (128, 128)
    batch_size = 32
    epochs = 10

    print("Loading dataset...")
    train_ds, val_ds, class_names = load_data(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size
    )

    num_classes = len(class_names)
    input_shape = (128, 128, 3)

    print(f"Classes found: {num_classes}")

    # Select model
    if model_name == "mlp":
        model = create_mlp_model(input_shape, num_classes)

    elif model_name == "cnn":
        model = create_cnn_model(input_shape, num_classes)

    elif model_name == "transfer":
        model = create_transfer_learning_model(input_shape, num_classes)

    else:
        raise ValueError("Model must be: mlp | cnn | transfer")

    # Compile
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # Save model
    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = save_dir / f"{model_name}_{timestamp}.keras"

    model.save(model_path)

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cnn",
                        help="mlp | cnn | transfer")
    args = parser.parse_args()

    train_model(args.model)
