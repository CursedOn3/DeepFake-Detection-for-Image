"""
Simple Script to Check if an Image is a Deepfake
Usage: python check_deepfake.py path/to/your/image.jpg
"""

import sys
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import argparse

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_model(model_path='models/deepfake_detector.h5'):
    """Load the trained deepfake detection model"""
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("Please train a model first using: python main.py --train")
        sys.exit(1)
    
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("✓ Model loaded successfully!\n")
    return model

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for prediction"""
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        sys.exit(1)
    
    # Load and resize image
    img = Image.open(image_path)
    img = img.convert('RGB')  # Ensure RGB format
    img = img.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_deepfake(model, image_path):
    """Predict if an image is a deepfake"""
    # Preprocess image
    img_array = preprocess_image(image_path)
    
    # Make prediction
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # Interpret results
    is_fake = prediction > 0.5
    confidence = prediction if is_fake else (1 - prediction)
    label = "FAKE (Deepfake)" if is_fake else "REAL (Authentic)"
    
    return {
        'label': label,
        'is_fake': is_fake,
        'confidence': confidence * 100,
        'raw_score': prediction
    }

def print_result(image_path, result):
    """Print formatted result"""
    filename = os.path.basename(image_path)
    
    print("=" * 70)
    print("                    DEEPFAKE DETECTION RESULT")
    print("=" * 70)
    print(f"\n📁 Image: {filename}")
    print(f"📍 Path: {image_path}")
    print("\n" + "-" * 70)
    
    # Color-coded result
    if result['is_fake']:
        status_icon = "⚠️"
        status_color = "FAKE"
    else:
        status_icon = "✅"
        status_color = "REAL"
    
    print(f"\n{status_icon}  Prediction: {result['label']}")
    print(f"📊 Confidence: {result['confidence']:.2f}%")
    print(f"🔢 Raw Score: {result['raw_score']:.4f} (>0.5 = Fake, <0.5 = Real)")
    
    print("\n" + "-" * 70)
    
    # Interpretation guide
    print("\n💡 Interpretation Guide:")
    if result['confidence'] > 90:
        print("   → Very High Confidence - Result is highly reliable")
    elif result['confidence'] > 75:
        print("   → High Confidence - Result is reliable")
    elif result['confidence'] > 60:
        print("   → Moderate Confidence - Result is fairly reliable")
    else:
        print("   → Low Confidence - Result is uncertain, manual review recommended")
    
    print("\n" + "=" * 70)

def check_multiple_images(model, image_paths):
    """Check multiple images"""
    results = []
    
    print(f"\n🔍 Checking {len(image_paths)} image(s)...\n")
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Processing: {os.path.basename(image_path)}")
        
        try:
            result = predict_deepfake(model, image_path)
            results.append({
                'path': image_path,
                'filename': os.path.basename(image_path),
                **result
            })
            print(f"    Result: {result['label']} ({result['confidence']:.2f}%)")
        except Exception as e:
            print(f"    ❌ Error: {str(e)}")
            results.append({
                'path': image_path,
                'filename': os.path.basename(image_path),
                'error': str(e)
            })
        print()
    
    return results

def print_batch_summary(results):
    """Print summary of batch results"""
    total = len(results)
    errors = sum(1 for r in results if 'error' in r)
    successful = total - errors
    
    if successful == 0:
        print("❌ All images failed to process")
        return
    
    fake_count = sum(1 for r in results if 'is_fake' in r and r['is_fake'])
    real_count = successful - fake_count
    
    print("\n" + "=" * 70)
    print("                        BATCH SUMMARY")
    print("=" * 70)
    print(f"\n📊 Total Images Processed: {successful}/{total}")
    print(f"✅ Real (Authentic): {real_count} ({real_count/successful*100:.1f}%)")
    print(f"⚠️  Fake (Deepfake): {fake_count} ({fake_count/successful*100:.1f}%)")
    
    if errors > 0:
        print(f"❌ Errors: {errors}")
    
    print("\n" + "-" * 70)
    print("\nDetailed Results:")
    print("-" * 70)
    
    for i, result in enumerate(results, 1):
        if 'error' in result:
            print(f"{i}. {result['filename']}: ❌ ERROR - {result['error']}")
        else:
            icon = "⚠️" if result['is_fake'] else "✅"
            print(f"{i}. {result['filename']}: {icon} {result['label']} ({result['confidence']:.2f}%)")
    
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(
        description='Check if an image is a deepfake',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check a single image
  python check_deepfake.py photo.jpg
  
  # Check multiple images
  python check_deepfake.py photo1.jpg photo2.jpg photo3.jpg
  
  # Use custom model
  python check_deepfake.py photo.jpg --model models/my_model.h5
        """
    )
    
    parser.add_argument('images', nargs='+', help='Path(s) to image file(s)')
    parser.add_argument('--model', default='models/deepfake_detector.h5',
                        help='Path to trained model (default: models/deepfake_detector.h5)')
    parser.add_argument('--batch', action='store_true',
                        help='Show batch summary for multiple images')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 70)
    print("           🔍 DEEPFAKE DETECTION SYSTEM 🔍")
    print("=" * 70)
    
    # Load model
    model = load_model(args.model)
    
    # Process images
    if len(args.images) == 1 and not args.batch:
        # Single image - detailed output
        result = predict_deepfake(model, args.images[0])
        print_result(args.images[0], result)
    else:
        # Multiple images - batch processing
        results = check_multiple_images(model, args.images)
        print_batch_summary(results)
    
    print("\n✅ Analysis complete!\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
