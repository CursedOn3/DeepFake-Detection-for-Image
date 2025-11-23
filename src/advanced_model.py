"""
Advanced Deep Fake Detection Model
Inspired by ResNext architecture with Attention mechanisms for images
Similar to abhijithjadhav's approach but adapted for static images
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50, EfficientNetB0
import numpy as np
from . import config


class AttentionLayer(layers.Layer):
    """
    Custom attention layer for focusing on important image regions
    """
    def __init__(self, units=256, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_W'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='attention_b'
        )
        self.u = self.add_weight(
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_u'
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # x shape: (batch_size, height, width, channels)
        # Reshape to (batch_size, h*w, channels)
        shape = tf.shape(x)
        batch_size, h, w, c = shape[0], shape[1], shape[2], shape[3]
        
        x_reshape = tf.reshape(x, [batch_size, h * w, c])
        
        # Calculate attention scores
        uit = tf.tanh(tf.matmul(x_reshape, self.W) + self.b)
        ait = tf.matmul(uit, tf.expand_dims(self.u, -1))
        ait = tf.squeeze(ait, -1)
        ait = tf.nn.softmax(ait, axis=-1)
        
        # Apply attention weights
        ait = tf.expand_dims(ait, -1)
        weighted = x_reshape * ait
        
        # Sum over spatial dimensions
        output = tf.reduce_sum(weighted, axis=1)
        
        return output
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({"units": self.units})
        return config


class ResNextBasedModel:
    """
    ResNext-inspired model for deepfake detection
    Similar architecture to the reference project but for images
    """
    
    @staticmethod
    def create_model(input_shape=(224, 224, 3), num_classes=2):
        """
        Create ResNext-based model with attention mechanism
        
        Args:
            input_shape: Input image shape
            num_classes: Number of output classes (2 for binary classification)
        
        Returns:
            Compiled Keras model
        """
        print("\n" + "="*60)
        print("CREATING RESNEXT-INSPIRED MODEL")
        print("="*60)
        
        inputs = layers.Input(shape=input_shape, name='input_image')
        
        # Use ResNet50 as backbone (similar to ResNext architecture)
        backbone = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
        
        # Freeze backbone initially for transfer learning
        backbone.trainable = False
        
        # Extract features
        x = backbone.output
        
        # Add attention mechanism
        attention_output = AttentionLayer(units=512, name='spatial_attention')(x)
        
        # Additional dense layers
        x = layers.Dense(512, activation='relu', name='fc1')(attention_output)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.Dropout(0.5, name='dropout1')(x)
        
        x = layers.Dense(256, activation='relu', name='fc2')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        x = layers.Dropout(0.4, name='dropout2')(x)
        
        # Output layer
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        else:
            outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='ResNext_Deepfake_Detector')
        
        print("✓ ResNext-based model created successfully")
        print(f"  - Backbone: ResNet50 (pretrained on ImageNet)")
        print(f"  - Attention mechanism: Spatial attention layer")
        print(f"  - Total parameters: {model.count_params():,}")
        
        return model
    
    @staticmethod
    def compile_model(model, learning_rate=0.001):
        """
        Compile model with appropriate loss and metrics
        
        Args:
            model: Keras model
            learning_rate: Learning rate for optimizer
        
        Returns:
            Compiled model
        """
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        print("✓ Model compiled successfully")
        print(f"  - Optimizer: Adam (lr={learning_rate})")
        print(f"  - Loss: binary_crossentropy")
        print(f"  - Metrics: accuracy, precision, recall, AUC")
        
        return model


class ImprovedEfficientNetModel:
    """
    Improved EfficientNet model with attention and better architecture
    """
    
    @staticmethod
    def create_model(input_shape=(224, 224, 3), num_classes=2):
        """
        Create improved EfficientNet model with attention
        
        Args:
            input_shape: Input image shape
            num_classes: Number of output classes
        
        Returns:
            Compiled Keras model
        """
        print("\n" + "="*60)
        print("CREATING IMPROVED EFFICIENTNET MODEL")
        print("="*60)
        
        inputs = layers.Input(shape=input_shape, name='input_image')
        
        # EfficientNet backbone
        backbone = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
        
        backbone.trainable = False
        
        # Get features
        x = backbone.output
        
        # Add attention
        attention_output = AttentionLayer(units=512, name='spatial_attention')(x)
        
        # Classification head
        x = layers.Dense(512, activation='relu', name='fc1')(attention_output)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.Dropout(0.5, name='dropout1')(x)
        
        x = layers.Dense(256, activation='relu', name='fc2')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        x = layers.Dropout(0.4, name='dropout2')(x)
        
        # Output
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        else:
            outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='EfficientNet_Attention_Detector')
        
        print("✓ Improved EfficientNet model created successfully")
        print(f"  - Backbone: EfficientNetB0 (pretrained on ImageNet)")
        print(f"  - Attention mechanism: Spatial attention layer")
        print(f"  - Total parameters: {model.count_params():,}")
        
        return model
    
    @staticmethod
    def compile_model(model, learning_rate=0.001):
        """Compile the model"""
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        print("✓ Model compiled successfully")
        return model


def create_advanced_model(model_type='resnext', input_shape=(224, 224, 3)):
    """
    Factory function to create and compile advanced models
    
    Args:
        model_type: 'resnext' or 'efficientnet'
        input_shape: Input image shape
    
    Returns:
        Compiled Keras model
    """
    if model_type.lower() == 'resnext':
        model = ResNextBasedModel.create_model(input_shape=input_shape)
        model = ResNextBasedModel.compile_model(model)
    elif model_type.lower() == 'efficientnet':
        model = ImprovedEfficientNetModel.create_model(input_shape=input_shape)
        model = ImprovedEfficientNetModel.compile_model(model)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'resnext' or 'efficientnet'")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing ResNext model creation...")
    resnext_model = create_advanced_model('resnext')
    resnext_model.summary()
    
    print("\n" + "="*60)
    print("\nTesting EfficientNet model creation...")
    efficientnet_model = create_advanced_model('efficientnet')
    efficientnet_model.summary()
