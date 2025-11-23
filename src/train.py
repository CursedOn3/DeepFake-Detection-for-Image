"""
Training Script for Deepfake Detection Model
Handles model training with data generators, callbacks, and validation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, 
    TensorBoard, CSVLogger
)

from . import config
from .model import DeepfakeDetector
from .preprocess import ImagePreprocessor
from .advanced_model import create_advanced_model


class ModelTrainer:
    """
    Handles the complete training pipeline for deepfake detection models
    """
    
    def __init__(self, model_type=config.MODEL_TYPE):
        """
        Initialize the trainer
        
        Args:
            model_type (str): Type of model to train
        """
        self.model_type = model_type
        self.model = None
        self.history = None
        self.preprocessor = ImagePreprocessor()
        
        # Set random seeds for reproducibility
        np.random.seed(config.RANDOM_SEED)
        tf.random.set_seed(config.RANDOM_SEED)
    
    def setup_callbacks(self):
        """
        Setup training callbacks
        
        Returns:
            list: List of Keras callbacks
        """
        print("\nSetting up training callbacks...")
        
        callbacks = []
        
        # Model checkpoint - save best model
        checkpoint = ModelCheckpoint(
            filepath=config.MODEL_CHECKPOINT_PATH,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard logging
        log_dir = os.path.join(config.RESULTS_DIR, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard)
        
        # CSV Logger
        csv_logger = CSVLogger(
            os.path.join(config.RESULTS_DIR, 'training_log.csv'),
            append=True
        )
        callbacks.append(csv_logger)
        
        print("✓ Callbacks configured:")
        print("  - Model checkpoint (best model)")
        print("  - Early stopping")
        print("  - Learning rate reduction")
        print("  - TensorBoard logging")
        print("  - CSV logging")
        
        return callbacks
    
    def train_with_generators(self, train_dir=config.TRAIN_DIR, val_dir=config.VAL_DIR, 
                             epochs=config.EPOCHS, batch_size=config.BATCH_SIZE):
        """
        Train model using data generators
        
        Args:
            train_dir (str): Training data directory
            val_dir (str): Validation data directory
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            
        Returns:
            History: Training history object
        """
        print("\n" + "="*60)
        print("STARTING MODEL TRAINING")
        print("="*60)
        
        # Create model
        print("\nCreating model...")
        self.model = DeepfakeDetector.get_model(self.model_type)
        
        # Create data generators
        print("\nCreating data generators...")
        train_generator, val_generator = self.preprocessor.create_data_generators(
            train_dir, val_dir, batch_size
        )
        
        print(f"\n✓ Training samples: {train_generator.samples}")
        print(f"✓ Validation samples: {val_generator.samples}")
        print(f"✓ Batch size: {batch_size}")
        print(f"✓ Steps per epoch: {train_generator.samples // batch_size}")
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Train model
        print("\n" + "="*60)
        print("TRAINING IN PROGRESS")
        print("="*60 + "\n")
        
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=config.VERBOSE_TRAINING
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        
        # Save final model
        self.save_model(config.MODEL_SAVE_PATH)
        
        return self.history
    
    def train_with_arrays(self, X_train, y_train, X_val, y_val, 
                         epochs=config.EPOCHS, batch_size=config.BATCH_SIZE):
        """
        Train model using numpy arrays
        
        Args:
            X_train (numpy.ndarray): Training images
            y_train (numpy.ndarray): Training labels
            X_val (numpy.ndarray): Validation images
            y_val (numpy.ndarray): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            
        Returns:
            History: Training history object
        """
        print("\n" + "="*60)
        print("STARTING MODEL TRAINING (Array Mode)")
        print("="*60)
        
        # Create model
        print("\nCreating model...")
        self.model = DeepfakeDetector.get_model(self.model_type)
        
        print(f"\n✓ Training samples: {len(X_train)}")
        print(f"✓ Validation samples: {len(X_val)}")
        print(f"✓ Training shape: {X_train.shape}")
        print(f"✓ Validation shape: {X_val.shape}")
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Train model
        print("\n" + "="*60)
        print("TRAINING IN PROGRESS")
        print("="*60 + "\n")
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=config.VERBOSE_TRAINING
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        
        # Save final model
        self.save_model(config.MODEL_SAVE_PATH)
        
        return self.history
    
    def fine_tune(self, train_dir=config.TRAIN_DIR, val_dir=config.VAL_DIR,
                  epochs=10, initial_learning_rate=1e-5):
        """
        Fine-tune a pre-trained model by unfreezing some layers
        
        Args:
            train_dir (str): Training data directory
            val_dir (str): Validation data directory
            epochs (int): Number of fine-tuning epochs
            initial_learning_rate (float): Learning rate for fine-tuning
        """
        if self.model is None:
            print("Error: No model loaded. Train or load a model first.")
            return
        
        print("\n" + "="*60)
        print("STARTING FINE-TUNING")
        print("="*60)
        
        # Unfreeze model layers
        if self.model_type == 'efficientnet':
            self.model = DeepfakeDetector.unfreeze_model(self.model, num_layers_to_unfreeze=30)
        
        # Recompile with lower learning rate
        self.model = DeepfakeDetector.compile_model(self.model, learning_rate=initial_learning_rate)
        
        # Create data generators
        train_generator, val_generator = self.preprocessor.create_data_generators(
            train_dir, val_dir, config.BATCH_SIZE
        )
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Fine-tune
        print("\nFine-tuning model...")
        fine_tune_history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=config.VERBOSE_TRAINING
        )
        
        # Append to history
        if self.history is not None:
            for key in self.history.history.keys():
                self.history.history[key].extend(fine_tune_history.history[key])
        else:
            self.history = fine_tune_history
        
        # Save fine-tuned model
        self.save_model(config.MODEL_SAVE_PATH.replace('.h5', '_finetuned.h5'))
        
        print("\n✓ Fine-tuning completed")
    
    def save_model(self, filepath):
        """
        Save the trained model
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            print("Error: No model to save")
            return
        
        self.model.save(filepath)
        print(f"\n✓ Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model
        
        Args:
            filepath (str): Path to the model file
        """
        if not os.path.exists(filepath):
            print(f"Error: Model file not found: {filepath}")
            return
        
        self.model = tf.keras.models.load_model(filepath)
        print(f"✓ Model loaded from: {filepath}")
    
    def plot_training_history(self, save_path=None):
        """
        Plot and save training history
        
        Args:
            save_path (str): Path to save the plot (if None, uses config path)
        """
        if self.history is None:
            print("Error: No training history available")
            return
        
        if save_path is None:
            save_path = config.TRAINING_HISTORY_PATH
        
        history = self.history.history
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot accuracy
        axes[0, 0].plot(history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot loss
        axes[0, 1].plot(history['loss'], label='Train Loss')
        axes[0, 1].plot(history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot precision
        if 'precision' in history:
            axes[1, 0].plot(history['precision'], label='Train Precision')
            axes[1, 0].plot(history['val_precision'], label='Val Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot recall
        if 'recall' in history:
            axes[1, 1].plot(history['recall'], label='Train Recall')
            axes[1, 1].plot(history['val_recall'], label='Val Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✓ Training history plot saved to: {save_path}")
    
    def print_training_summary(self):
        """Print summary of training results"""
        if self.history is None:
            print("Error: No training history available")
            return
        
        history = self.history.history
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        final_epoch = len(history['loss'])
        
        print(f"\nTotal epochs trained: {final_epoch}")
        print("\nFinal Metrics:")
        print(f"  Training Loss:     {history['loss'][-1]:.4f}")
        print(f"  Training Accuracy: {history['accuracy'][-1]:.4f}")
        print(f"  Validation Loss:     {history['val_loss'][-1]:.4f}")
        print(f"  Validation Accuracy: {history['val_accuracy'][-1]:.4f}")
        
        if 'precision' in history:
            print(f"  Training Precision:   {history['precision'][-1]:.4f}")
            print(f"  Validation Precision: {history['val_precision'][-1]:.4f}")
        
        if 'recall' in history:
            print(f"  Training Recall:   {history['recall'][-1]:.4f}")
            print(f"  Validation Recall: {history['val_recall'][-1]:.4f}")
        
        print("\nBest Validation Accuracy: {:.4f} (Epoch {})".format(
            max(history['val_accuracy']),
            history['val_accuracy'].index(max(history['val_accuracy'])) + 1
        ))
        
        print("="*60 + "\n")


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Deepfake Detection Model')
    parser.add_argument('--model', type=str, default=config.MODEL_TYPE,
                       choices=['efficientnet', 'custom_cnn', 'deep_cnn'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--fine_tune', action='store_true',
                       help='Fine-tune after initial training')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ModelTrainer(model_type=args.model)
    
    # Train model
    history = trainer.train_with_generators(
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Print summary
    trainer.print_training_summary()
    
    # Plot history
    trainer.plot_training_history()
    
    # Fine-tune if requested
    if args.fine_tune and args.model == 'efficientnet':
        print("\nStarting fine-tuning phase...")
        trainer.fine_tune(epochs=10)
        trainer.print_training_summary()
        trainer.plot_training_history()


if __name__ == "__main__":
    main()
