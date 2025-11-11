"""
Configuration file for Deepfake Detection System
Contains all hyperparameters and path configurations
"""

import os

# ==================== PATHS ====================
# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Data subdirectories (create these manually and add your datasets)
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Dataset structure: 
# data/train/real/*.jpg
# data/train/fake/*.jpg
# data/val/real/*.jpg
# data/val/fake/*.jpg
# data/test/real/*.jpg
# data/test/fake/*.jpg

# Model paths
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'deepfake_detector.h5')
MODEL_CHECKPOINT_PATH = os.path.join(MODELS_DIR, 'checkpoint_best.h5')

# Results paths
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
TRAINING_HISTORY_PATH = os.path.join(RESULTS_DIR, 'training_history.png')
CLASSIFICATION_REPORT_PATH = os.path.join(RESULTS_DIR, 'classification_report.txt')

# ==================== MODEL CONFIGURATION ====================
# Model architecture choice: 'efficientnet' or 'custom_cnn'
MODEL_TYPE = 'efficientnet'

# Image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Class configuration
NUM_CLASSES = 1  # Binary classification (sigmoid output)
CLASS_NAMES = ['real', 'fake']

# ==================== TRAINING HYPERPARAMETERS ====================
# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Data split ratios (if splitting from single directory)
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Early stopping parameters
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# ==================== DATA AUGMENTATION ====================
# Augmentation parameters for training
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.15,
    'zoom_range': 0.15,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# ==================== FACE EXTRACTION ====================
# Face detection configuration
USE_FACE_EXTRACTION = False  # Set to True to extract faces before classification
FACE_DETECTION_METHOD = 'mtcnn'  # Options: 'mtcnn', 'opencv_haar', 'opencv_dnn'
FACE_CONFIDENCE_THRESHOLD = 0.9  # Minimum confidence for face detection

# MTCNN configuration
MTCNN_MIN_FACE_SIZE = 40
MTCNN_SCALE_FACTOR = 0.709

# OpenCV Haar Cascade path (if using opencv_haar method)
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# ==================== INFERENCE CONFIGURATION ====================
# Classification threshold
CLASSIFICATION_THRESHOLD = 0.5  # Probabilities >= 0.5 are classified as 'fake'

# Batch processing
INFERENCE_BATCH_SIZE = 16

# ==================== LOGGING ====================
# Verbose modes
VERBOSE_TRAINING = 1  # 0 = silent, 1 = progress bar, 2 = one line per epoch
VERBOSE_EVALUATION = 1

# Random seed for reproducibility
RANDOM_SEED = 42

# ==================== HELPER FUNCTIONS ====================
def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        DATA_DIR, MODELS_DIR, RESULTS_DIR,
        TRAIN_DIR, VAL_DIR, TEST_DIR,
        os.path.join(TRAIN_DIR, 'real'),
        os.path.join(TRAIN_DIR, 'fake'),
        os.path.join(VAL_DIR, 'real'),
        os.path.join(VAL_DIR, 'fake'),
        os.path.join(TEST_DIR, 'real'),
        os.path.join(TEST_DIR, 'fake')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ All required directories created successfully!")

def print_config():
    """Print current configuration"""
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION SYSTEM - CONFIGURATION")
    print("="*60)
    print(f"Model Type: {MODEL_TYPE}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Face Extraction: {'Enabled' if USE_FACE_EXTRACTION else 'Disabled'}")
    print(f"Random Seed: {RANDOM_SEED}")
    print("="*60 + "\n")

if __name__ == "__main__":
    create_directories()
    print_config()
