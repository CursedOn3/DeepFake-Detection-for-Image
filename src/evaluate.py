"""
Model Evaluation Script for Deepfake Detection
Evaluates model performance with various metrics and visualizations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_curve, auc, roc_auc_score
)
import tensorflow as tf

from . import config
from .preprocess import ImagePreprocessor


class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics and visualizations
    """
    
    def __init__(self, model_path=config.MODEL_SAVE_PATH):
        """
        Initialize evaluator
        
        Args:
            model_path (str): Path to the trained model
        """
        self.model_path = model_path
        self.model = None
        self.preprocessor = ImagePreprocessor()
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            print(f"Error: Model file not found: {self.model_path}")
            return
        
        print(f"Loading model from: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
        print("✓ Model loaded successfully\n")
    
    def evaluate_with_generator(self, test_dir=config.TEST_DIR, batch_size=config.BATCH_SIZE):
        """
        Evaluate model using data generator
        
        Args:
            test_dir (str): Test data directory
            batch_size (int): Batch size for evaluation
            
        Returns:
            dict: Dictionary containing all metrics
        """
        if self.model is None:
            print("Error: No model loaded")
            return None
        
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60 + "\n")
        
        # Create test generator
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=config.IMG_SIZE,
            batch_size=batch_size,
            class_mode='binary',
            classes=config.CLASS_NAMES,
            shuffle=False
        )
        
        print(f"Test samples: {test_generator.samples}")
        print(f"Test steps: {len(test_generator)}\n")
        
        # Get predictions
        print("Generating predictions...")
        y_pred_probs = self.model.predict(test_generator, verbose=1)
        y_pred = (y_pred_probs > config.CLASSIFICATION_THRESHOLD).astype(int).flatten()
        y_true = test_generator.classes
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_probs)
        
        # Print metrics
        self.print_metrics(metrics)
        
        # Generate visualizations
        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_roc_curve(y_true, y_pred_probs)
        self.save_classification_report(y_true, y_pred)
        
        return metrics
    
    def evaluate_with_arrays(self, X_test, y_test):
        """
        Evaluate model using numpy arrays
        
        Args:
            X_test (numpy.ndarray): Test images
            y_test (numpy.ndarray): Test labels
            
        Returns:
            dict: Dictionary containing all metrics
        """
        if self.model is None:
            print("Error: No model loaded")
            return None
        
        print("\n" + "="*60)
        print("EVALUATING MODEL (Array Mode)")
        print("="*60 + "\n")
        
        print(f"Test samples: {len(X_test)}")
        print(f"Test shape: {X_test.shape}\n")
        
        # Get predictions
        print("Generating predictions...")
        y_pred_probs = self.model.predict(X_test, batch_size=config.BATCH_SIZE, verbose=1)
        y_pred = (y_pred_probs > config.CLASSIFICATION_THRESHOLD).astype(int).flatten()
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_probs)
        
        # Print metrics
        self.print_metrics(metrics)
        
        # Generate visualizations
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_roc_curve(y_test, y_pred_probs)
        self.save_classification_report(y_test, y_pred)
        
        return metrics
    
    def calculate_metrics(self, y_true, y_pred, y_pred_probs):
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true (numpy.ndarray): True labels
            y_pred (numpy.ndarray): Predicted labels
            y_pred_probs (numpy.ndarray): Prediction probabilities
            
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y_true, y_pred_probs)
        }
        
        return metrics
    
    def print_metrics(self, metrics):
        """
        Print evaluation metrics in a formatted way
        
        Args:
            metrics (dict): Dictionary of metrics
        """
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        
        print(f"\nAccuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print("\n" + "="*60 + "\n")
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Plot and save confusion matrix
        
        Args:
            y_true (numpy.ndarray): True labels
            y_pred (numpy.ndarray): Predicted labels
            save_path (str): Path to save the plot
        """
        if save_path is None:
            save_path = config.CONFUSION_MATRIX_PATH
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=config.CLASS_NAMES,
                   yticklabels=config.CLASS_NAMES,
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add accuracy to plot
        accuracy = np.trace(cm) / np.sum(cm)
        plt.text(1, -0.3, f'Accuracy: {accuracy:.2%}', 
                fontsize=12, ha='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Confusion matrix saved to: {save_path}")
    
    def plot_roc_curve(self, y_true, y_pred_probs, save_path=None):
        """
        Plot and save ROC curve
        
        Args:
            y_true (numpy.ndarray): True labels
            y_pred_probs (numpy.ndarray): Prediction probabilities
            save_path (str): Path to save the plot
        """
        if save_path is None:
            save_path = os.path.join(config.RESULTS_DIR, 'roc_curve.png')
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
        roc_auc = auc(fpr, tpr)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ ROC curve saved to: {save_path}")
    
    def save_classification_report(self, y_true, y_pred, save_path=None):
        """
        Save detailed classification report to file
        
        Args:
            y_true (numpy.ndarray): True labels
            y_pred (numpy.ndarray): Predicted labels
            save_path (str): Path to save the report
        """
        if save_path is None:
            save_path = config.CLASSIFICATION_REPORT_PATH
        
        # Generate classification report
        report = classification_report(
            y_true, y_pred,
            target_names=config.CLASS_NAMES,
            digits=4
        )
        
        # Save to file
        with open(save_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(report)
            f.write("\n" + "="*60 + "\n")
        
        print(f"✓ Classification report saved to: {save_path}")
        
        # Also print to console
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60 + "\n")
        print(report)
    
    def predict_sample_images(self, image_paths, visualize=True):
        """
        Predict on sample images and optionally visualize results
        
        Args:
            image_paths (list): List of image paths
            visualize (bool): Whether to visualize predictions
            
        Returns:
            list: List of predictions
        """
        if self.model is None:
            print("Error: No model loaded")
            return None
        
        predictions = []
        
        for img_path in image_paths:
            # Preprocess image
            img = self.preprocessor.preprocess_for_inference(img_path)
            
            if img is None:
                print(f"Error processing: {img_path}")
                continue
            
            # Predict
            pred_prob = self.model.predict(img, verbose=0)[0][0]
            pred_class = 'fake' if pred_prob >= config.CLASSIFICATION_THRESHOLD else 'real'
            
            predictions.append({
                'path': img_path,
                'probability': pred_prob,
                'class': pred_class,
                'confidence': pred_prob if pred_class == 'fake' else 1 - pred_prob
            })
        
        # Visualize if requested
        if visualize and len(predictions) > 0:
            self.visualize_predictions(predictions)
        
        return predictions
    
    def visualize_predictions(self, predictions, max_images=6):
        """
        Visualize predictions on sample images
        
        Args:
            predictions (list): List of prediction dictionaries
            max_images (int): Maximum number of images to display
        """
        import cv2
        
        n_images = min(len(predictions), max_images)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, pred in enumerate(predictions[:n_images]):
            # Load and display image
            img = cv2.imread(pred['path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(img)
            
            # Set title with prediction
            color = 'red' if pred['class'] == 'fake' else 'green'
            title = f"Predicted: {pred['class'].upper()}\n"
            title += f"Confidence: {pred['confidence']:.2%}"
            
            axes[i].set_title(title, fontsize=12, color=color, fontweight='bold')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, 'sample_predictions.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Prediction visualization saved to {config.RESULTS_DIR}")


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Deepfake Detection Model')
    parser.add_argument('--model_path', type=str, default=config.MODEL_SAVE_PATH,
                       help='Path to trained model')
    parser.add_argument('--test_dir', type=str, default=config.TEST_DIR,
                       help='Test data directory')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(model_path=args.model_path)
    
    # Evaluate model
    if evaluator.model is not None:
        metrics = evaluator.evaluate_with_generator(
            test_dir=args.test_dir,
            batch_size=args.batch_size
        )
        
        print("\n✓ Evaluation complete!")
        print(f"  Results saved to: {config.RESULTS_DIR}")
    else:
        print("\nError: Could not load model for evaluation")


if __name__ == "__main__":
    main()
