"""
Improved Training Pipeline for Plant Disease Detection
Uses EfficientNetB0 transfer learning with proper callbacks and class weights
"""

import os
import sys
import tensorflow as tf
from datetime import datetime
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader_v2 import load_data_with_augmentation, calculate_class_weights
from src.models.efficientnet_transfer import create_efficientnet_model, compile_model, unfreeze_and_compile_finetuning


def train_improved_model(epochs_phase1=20, epochs_phase2=15):
    """
    Train plant disease detection model with two-phase training:
    Phase 1: Train with frozen base model
    Phase 2: Fine-tune with unfrozen layers
    
    Args:
        epochs_phase1: Epochs for phase 1 (frozen base)
        epochs_phase2: Epochs for phase 2 (fine-tuning)
    """
    
    print("\n" + "=" * 70)
    print("🌿 PLANT DISEASE DETECTION - IMPROVED TRAINING PIPELINE")
    print("=" * 70)
    
    # Configuration
    data_dir = "data/raw"
    img_size = (224, 224)
    batch_size = 32
    
    print("\n📊 PHASE 1: LOADING DATA AND CALCULATING CLASS WEIGHTS")
    print("-" * 70)
    
    try:
        # Load data with augmentation
        print("Loading training and validation datasets...")
        train_ds, val_ds, class_names = load_data_with_augmentation(
            data_dir=data_dir,
            img_size=img_size,
            batch_size=batch_size
        )
        
        num_classes = len(class_names)
        print(f"✓ Datasets loaded successfully!")
        print(f"  - Classes: {num_classes}")
        print(f"  - Image size: {img_size}")
        print(f"  - Batch size: {batch_size}")
        print(f"\nClasses: {', '.join(class_names[:5])}...")
        
        # Calculate class weights for imbalanced dataset
        print("\nCalculating class weights for imbalanced dataset...")
        class_weights, class_counts = calculate_class_weights(data_dir)
        print(f"✓ Class weights calculated!")
        print(f"  - Min weight: {min(class_weights.values()):.3f}")
        print(f"  - Max weight: {max(class_weights.values()):.3f}")
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🏗️  PHASE 2: BUILDING MODEL")
    print("-" * 70)
    
    try:
        # Create model with frozen base
        print("Creating EfficientNetB0 transfer learning model...")
        model, base_model = create_efficientnet_model(
            input_shape=(224, 224, 3),
            num_classes=num_classes,
            dropout_rate=0.4
        )
        
        # Compile for initial training
        model = compile_model(model, learning_rate=0.001)
        
        print("✓ Model created successfully!")
        print(f"  - Base model: EfficientNetB0")
        print(f"  - Total parameters: {model.count_params():,}")
        print(f"  - Trainable parameters (phase 1): {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
        
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎓 PHASE 3: TRAINING WITH FROZEN BASE MODEL")
    print("-" * 70)
    print(f"Training for {epochs_phase1} epochs with frozen base model...")
    
    try:
        # Callbacks for phase 1
        callbacks_phase1 = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='saved_models/best_model_phase1.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0
            ),
        ]
        
        # Train phase 1
        history_phase1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_phase1,
            class_weight=class_weights,
            callbacks=callbacks_phase1,
            verbose=1
        )
        
        print("✓ Phase 1 training completed!")
        best_val_acc_p1 = max(history_phase1.history['val_accuracy'])
        print(f"  - Best validation accuracy: {best_val_acc_p1:.4f}")
        
    except Exception as e:
        print(f"✗ Error in phase 1 training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🔓 PHASE 4: UNFREEZING BASE MODEL FOR FINE-TUNING")
    print("-" * 70)
    
    try:
        # Unfreeze base model and recompile
        print("Unfreezing last 50 layers of EfficientNetB0...")
        model = unfreeze_and_compile_finetuning(
            model,
            num_layers_to_unfreeze=50,
            learning_rate=0.0001
        )
        
        trainable_params_p2 = sum([tf.size(w).numpy() for w in model.trainable_weights])
        print(f"✓ Model recompiled for fine-tuning!")
        print(f"  - Trainable parameters (phase 2): {trainable_params_p2:,}")
        
    except Exception as e:
        print(f"✗ Error unfreezing model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎓 PHASE 5: FINE-TUNING WITH UNFROZEN LAYERS")
    print("-" * 70)
    print(f"Fine-tuning for {epochs_phase2} epochs with unfrozen layers...")
    
    try:
        # Callbacks for phase 2
        callbacks_phase2 = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='saved_models/best_model_phase2.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0
            ),
        ]
        
        # Train phase 2
        history_phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_phase2,
            class_weight=class_weights,
            callbacks=callbacks_phase2,
            verbose=1
        )
        
        print("✓ Phase 2 fine-tuning completed!")
        best_val_acc_p2 = max(history_phase2.history['val_accuracy'])
        print(f"  - Best validation accuracy: {best_val_acc_p2:.4f}")
        
    except Exception as e:
        print(f"✗ Error in phase 2 training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n💾 PHASE 6: SAVING MODEL AND METADATA")
    print("-" * 70)
    
    try:
        # Create save directory
        save_dir = Path("saved_models")
        save_dir.mkdir(exist_ok=True)
        
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = save_dir / f"efficientnet_model_{timestamp}.h5"
        model.save(str(final_model_path))
        print(f"✓ Model saved: {final_model_path}")
        
        # Save model as SavedModel format (newer)
        savedmodel_path = save_dir / f"efficientnet_savedmodel_{timestamp}"
        model.save(str(savedmodel_path))
        print(f"✓ Model saved (SavedModel): {savedmodel_path}")
        
        # Save class names
        class_names_path = save_dir / "class_names.json"
        with open(class_names_path, 'w') as f:
            json.dump(class_names, f, indent=2)
        print(f"✓ Class names saved: {class_names_path}")
        
        # Save training metadata
        metadata = {
            "model_type": "EfficientNetB0",
            "num_classes": num_classes,
            "image_size": img_size,
            "batch_size": batch_size,
            "phase1_epochs": epochs_phase1,
            "phase2_epochs": epochs_phase2,
            "best_val_accuracy_p1": float(best_val_acc_p1),
            "best_val_accuracy_p2": float(best_val_acc_p2),
            "timestamp": timestamp,
            "class_names": class_names,
            "class_weights": {str(k): float(v) for k, v in class_weights.items()}
        }
        
        metadata_path = save_dir / f"training_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved: {metadata_path}")
        
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nModel Performance:")
    print(f"  - Phase 1 Best Val Accuracy: {best_val_acc_p1:.4f}")
    print(f"  - Phase 2 Best Val Accuracy: {best_val_acc_p2:.4f}")
    print(f"  - Improvement: {(best_val_acc_p2 - best_val_acc_p1)*100:.2f}%")
    print(f"\nFinal Model Path: {final_model_path}")
    print(f"Classes: {num_classes}")
    print("=" * 70 + "\n")
    
    return True


if __name__ == "__main__":
    success = train_improved_model(epochs_phase1=20, epochs_phase2=15)
    sys.exit(0 if success else 1)
