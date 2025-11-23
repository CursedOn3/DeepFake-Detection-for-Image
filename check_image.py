"""
Simple Script to Check if an Image is Deepfake
Usage: python check_image.py <image_path>
"""

import sys
import os
import tensorflow as tf
from PIL import Image
import numpy as np

def load_model(model_path='models/deepfake_detector.h5'):
    """Load the trained model"""
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("   Please train a model first using:")
        print("   python main.py --train --model custom_cnn --epochs 10")
        sys.exit(1)
    
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("✓ Model loaded successfully!")
    return model

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for prediction"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        sys.exit(1)

def predict(model, image_array):
    """Make prediction"""
    prediction = model.predict(image_array, verbose=0)
    probability = float(prediction[0][0])
    
    # probability > 0.5 means FAKE, < 0.5 means REAL
    is_fake = probability > 0.5
    confidence = probability if is_fake else (1 - probability)
    
    return is_fake, confidence

def print_result(image_path, is_fake, confidence):
    """Print formatted result"""
    print("\n" + "="*60)
    print(f"IMAGE: {os.path.basename(image_path)}")
    print("="*60)
    
    if is_fake:
        print("🚨 PREDICTION: DEEPFAKE (AI-Generated/Manipulated)")
        emoji = "🚨"
    else:
        print("✅ PREDICTION: REAL (Authentic)")
        emoji = "✅"
    
    print(f"   Confidence: {confidence*100:.2f}%")
    
    # Confidence interpretation
    if confidence >= 0.9:
        print(f"   Certainty: Very High {emoji}")
    elif confidence >= 0.7:
        print(f"   Certainty: High")
    elif confidence >= 0.5:
        print(f"   Certainty: Moderate ⚠️")
    else:
        print(f"   Certainty: Low ⚠️ (uncertain)")
    
    print("="*60 + "\n")

def main():
    """Main function"""
    print("\n" + "="*60)
    print("         DEEPFAKE DETECTION - IMAGE CHECKER")
    print("="*60 + "\n")
    
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python check_image.py <image_path>")
        print("\nExamples:")
        print("  python check_image.py photo.jpg")
        print("  python check_image.py d:\\photos\\suspect.png")
        print("  python check_image.py \"C:\\Users\\Photos\\image.jpg\"")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        sys.exit(1)
    
    # Load model
    model = load_model()
    
    # Preprocess image
    print(f"Processing image: {image_path}")
    image_array = preprocess_image(image_path)
    
    # Make prediction
    print("Analyzing...")
    is_fake, confidence = predict(model, image_array)
    
    # Show result
    print_result(image_path, is_fake, confidence)

if __name__ == "__main__":
    main()
