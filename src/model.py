"""
Model Architecture Module for Deepfake Detection
Contains EfficientNetB0 and custom CNN architectures
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    BatchNormalization, GlobalAveragePooling2D, Input
)
import config


class DeepfakeDetector:
    """
    Factory class for creating deepfake detection models
    Supports EfficientNetB0 and custom CNN architectures
    """
    
    @staticmethod
    def create_efficientnet_model(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)):
        """
        Create a model based on EfficientNetB0 with transfer learning
        
        Args:
            input_shape (tuple): Input image shape (height, width, channels)
            
        Returns:
            keras.Model: Compiled model
        """
        print("\nBuilding EfficientNetB0 model...")
        
        # Input layer
        inputs = Input(shape=input_shape, name='input_layer')
        
        # Load pre-trained EfficientNetB0 (without top classification layer)
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs,
            pooling='avg'
        )
        
        # Freeze base model layers initially (for transfer learning)
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu', name='dense_1')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu', name='dense_2')(x)
        x = Dropout(0.2)(x)
        
        # Output layer (sigmoid for binary classification)
        outputs = Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs, name='EfficientNetB0_Deepfake_Detector')
        
        print("✓ EfficientNetB0 model created successfully")
        print(f"  Total layers: {len(model.layers)}")
        print(f"  Trainable: {sum([1 for layer in model.layers if layer.trainable])}")
        
        return model
    
    @staticmethod
    def create_custom_cnn_model(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)):
        """
        Create a custom CNN model inspired by MesoNet architecture
        Optimized for deepfake detection
        
        Args:
            input_shape (tuple): Input image shape (height, width, channels)
            
        Returns:
            keras.Model: Compiled model
        """
        print("\nBuilding Custom CNN model (MesoNet-inspired)...")
        
        model = models.Sequential(name='Custom_CNN_Deepfake_Detector')
        
        # Block 1
        model.add(Conv2D(8, (3, 3), padding='same', activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Block 2
        model.add(Conv2D(8, (5, 5), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Block 3
        model.add(Conv2D(16, (5, 5), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Block 4
        model.add(Conv2D(16, (5, 5), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(4, 4)))
        
        # Dense layers
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.5))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        print("✓ Custom CNN model created successfully")
        
        return model
    
    @staticmethod
    def create_deeper_cnn_model(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)):
        """
        Create a deeper custom CNN model for better feature extraction
        
        Args:
            input_shape (tuple): Input image shape (height, width, channels)
            
        Returns:
            keras.Model: Compiled model
        """
        print("\nBuilding Deeper Custom CNN model...")
        
        model = models.Sequential(name='Deep_CNN_Deepfake_Detector')
        
        # Block 1
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Block 2
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Block 3
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Block 4
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Global Average Pooling
        model.add(GlobalAveragePooling2D())
        
        # Dense layers
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        print("✓ Deeper Custom CNN model created successfully")
        
        return model
    
    @staticmethod
    def compile_model(model, learning_rate=config.LEARNING_RATE):
        """
        Compile the model with appropriate loss and optimizer
        
        Args:
            model (keras.Model): The model to compile
            learning_rate (float): Learning rate for optimizer
            
        Returns:
            keras.Model: Compiled model
        """
        print("\nCompiling model...")
        
        # Define optimizer
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        print("✓ Model compiled successfully")
        print(f"  Optimizer: Adam (lr={learning_rate})")
        print(f"  Loss: binary_crossentropy")
        print(f"  Metrics: accuracy, precision, recall, AUC")
        
        return model
    
    @staticmethod
    def get_model(model_type=config.MODEL_TYPE, input_shape=None):
        """
        Factory method to create and compile a model based on type
        
        Args:
            model_type (str): Type of model ('efficientnet', 'custom_cnn', 'deep_cnn', 'resnext', 'advanced_efficientnet')
            input_shape (tuple): Input shape (if None, uses config values)
            
        Returns:
            keras.Model: Compiled model ready for training
        """
        if input_shape is None:
            input_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
        
        print("\n" + "="*60)
        print(f"CREATING MODEL: {model_type.upper()}")
        print("="*60)
        
        # Create model based on type
        if model_type.lower() == 'efficientnet':
            model = DeepfakeDetector.create_efficientnet_model(input_shape)
        elif model_type.lower() == 'custom_cnn':
            model = DeepfakeDetector.create_custom_cnn_model(input_shape)
        elif model_type.lower() == 'deep_cnn':
            model = DeepfakeDetector.create_deeper_cnn_model(input_shape)
        elif model_type.lower() in ['resnext', 'advanced_efficientnet']:
            # Use advanced models with attention mechanism
            from .advanced_model import create_advanced_model
            model_name = 'resnext' if model_type.lower() == 'resnext' else 'efficientnet'
            model = create_advanced_model(model_name, input_shape)
            return model  # Already compiled in advanced_model.py
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Compile model
        model = DeepfakeDetector.compile_model(model)
        
        return model
    
    @staticmethod
    def unfreeze_model(model, num_layers_to_unfreeze=20):
        """
        Unfreeze the last N layers of the model for fine-tuning
        
        Args:
            model (keras.Model): Model to unfreeze
            num_layers_to_unfreeze (int): Number of layers to unfreeze from the end
            
        Returns:
            keras.Model: Model with unfrozen layers
        """
        print(f"\nUnfreezing last {num_layers_to_unfreeze} layers for fine-tuning...")
        
        # Unfreeze the specified number of layers from the end
        for layer in model.layers[-num_layers_to_unfreeze:]:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = True
        
        trainable_count = sum([1 for layer in model.layers if layer.trainable])
        print(f"✓ {trainable_count} layers are now trainable")
        
        return model


def print_model_summary(model):
    """
    Print detailed model summary
    
    Args:
        model (keras.Model): Model to summarize
    """
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60 + "\n")
    
    model.summary()
    
    print("\n" + "-"*60)
    print("MODEL STATISTICS")
    print("-"*60)
    
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print("-"*60 + "\n")


def test_model_creation():
    """Test model creation"""
    print("\n" + "="*60)
    print("TESTING MODEL CREATION")
    print("="*60 + "\n")
    
    # Test EfficientNet model
    try:
        print("\n1. Testing EfficientNetB0 model...")
        model_eff = DeepfakeDetector.get_model('efficientnet')
        print_model_summary(model_eff)
        print("✓ EfficientNetB0 model test passed")
    except Exception as e:
        print(f"✗ EfficientNetB0 model test failed: {e}")
    
    # Test Custom CNN model
    try:
        print("\n2. Testing Custom CNN model...")
        model_cnn = DeepfakeDetector.get_model('custom_cnn')
        print_model_summary(model_cnn)
        print("✓ Custom CNN model test passed")
    except Exception as e:
        print(f"✗ Custom CNN model test failed: {e}")
    
    # Test Deep CNN model
    try:
        print("\n3. Testing Deep CNN model...")
        model_deep = DeepfakeDetector.get_model('deep_cnn')
        print_model_summary(model_deep)
        print("✓ Deep CNN model test passed")
    except Exception as e:
        print(f"✗ Deep CNN model test failed: {e}")


if __name__ == "__main__":
    test_model_creation()
