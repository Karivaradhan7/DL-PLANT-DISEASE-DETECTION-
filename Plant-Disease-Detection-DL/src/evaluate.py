"""
Evaluation Module for Plant Disease Detection Models
Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Evaluate trained models"""
    
    def __init__(self, model, class_names, output_dir="../../outputs"):
        """
        Initialize evaluator
        
        Args:
            model: Trained model
            class_names: List of class names
            output_dir: Directory for saving outputs
        """
        self.model = model
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.y_true = None
        self.y_pred = None
        self.predictions = None
    
    def evaluate(self, test_dataset):
        """Evaluate model on test dataset"""
        # Get predictions and true labels
        self.y_true = []
        self.predictions = []
        
        for images, labels in test_dataset:
            batch_predictions = self.model.predict(images)
            self.predictions.extend(batch_predictions)
            self.y_true.extend(labels.numpy())
        
        self.y_pred = np.argmax(self.predictions, axis=1)
        self.y_true = np.array(self.y_true)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_true, self.y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        return metrics
    
    def get_classification_report(self):
        """Get detailed classification report"""
        report = classification_report(
            self.y_true, self.y_pred,
            target_names=self.class_names
        )
        return report
    
    def plot_confusion_matrix(self, normalize=True, figsize=(12, 10)):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, annot=True, fmt='.2f' if normalize else 'd',
            cmap='Blues', xticklabels=self.class_names,
            yticklabels=self.class_names, cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        save_path = self.output_dir / 'confusion_matrices' / 'confusion_matrix.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_top_predictions(self, num_images=9):
        """Plot top predictions"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.ravel()
        
        for idx in range(num_images):
            if idx < len(self.predictions):
                pred_class = self.y_pred[idx]
                true_class = self.y_true[idx]
                confidence = np.max(self.predictions[idx])
                
                axes[idx].set_title(
                    f"True: {self.class_names[true_class]}\n"
                    f"Pred: {self.class_names[pred_class]} ({confidence:.2f})",
                    color='green' if pred_class == true_class else 'red'
                )
                axes[idx].axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / 'plots' / 'top_predictions.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_class_accuracy(self):
        """Plot per-class accuracy"""
        class_accuracy = []
        
        for class_idx in range(len(self.class_names)):
            class_mask = self.y_true == class_idx
            if np.sum(class_mask) > 0:
                acc = np.sum(self.y_pred[class_mask] == class_idx) / np.sum(class_mask)
                class_accuracy.append(acc)
            else:
                class_accuracy.append(0)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(self.class_names, class_accuracy, color='skyblue', edgecolor='navy')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.ylabel('Accuracy')
        plt.title('Per-Class Accuracy')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = self.output_dir / 'plots' / 'class_accuracy.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_report(self):
        """Generate and save comprehensive evaluation report"""
        report_text = f"""
        ============================================
        Model Evaluation Report
        ============================================
        
        Classification Report:
        {self.get_classification_report()}
        
        """
        
        save_path = self.output_dir / 'evaluation_report.txt'
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        return save_path


if __name__ == "__main__":
    # Example usage
    print("Evaluation module loaded. Use with trained model and test dataset.")
