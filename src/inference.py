"""
Inference Script for Deepfake Detection
Classify new images as real or deepfake
"""

import os
import sys
import numpy as np
import tensorflow as tf
import cv2
from glob import glob

from . import config
from .preprocess import ImagePreprocessor


class DeepfakeInference:
    """
    Inference engine for deepfake detection on new images
    """
    
    def __init__(self, model_path=config.MODEL_SAVE_PATH):
        """
        Initialize inference engine
        
        Args:
            model_path (str): Path to trained model
        """
        self.model_path = model_path
        self.model = None
        self.preprocessor = ImagePreprocessor()
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            print(f"Error: Model file not found: {self.model_path}")
            print(f"Please train a model first using train.py")
            return False
        
        print(f"Loading model from: {self.model_path}")
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print("✓ Model loaded successfully\n")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_single_image(self, image_path, verbose=True):
        """
        Predict on a single image
        
        Args:
            image_path (str): Path to the image
            verbose (bool): Print results
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            print("Error: No model loaded")
            return None
        
        if not os.path.exists(image_path):
            print(f"Error: Image not found: {image_path}")
            return None
        
        # Preprocess image
        img = self.preprocessor.preprocess_for_inference(image_path)
        
        if img is None:
            print(f"Error: Could not process image: {image_path}")
            return None
        
        # Predict
        pred_prob = self.model.predict(img, verbose=0)[0][0]
        pred_class = 'fake' if pred_prob >= config.CLASSIFICATION_THRESHOLD else 'real'
        confidence = pred_prob if pred_class == 'fake' else 1 - pred_prob
        
        result = {
            'image_path': image_path,
            'prediction': pred_class,
            'probability': float(pred_prob),
            'confidence': float(confidence)
        }
        
        if verbose:
            self.print_prediction(result)
        
        return result
    
    def predict_batch(self, image_paths, batch_size=config.INFERENCE_BATCH_SIZE):
        """
        Predict on a batch of images
        
        Args:
            image_paths (list): List of image paths
            batch_size (int): Batch size for prediction
            
        Returns:
            list: List of prediction results
        """
        if self.model is None:
            print("Error: No model loaded")
            return None
        
        results = []
        total = len(image_paths)
        
        print(f"\nProcessing {total} images...")
        print("-" * 60)
        
        # Process in batches
        for i in range(0, total, batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_paths = []
            
            # Preprocess batch
            for img_path in batch_paths:
                if not os.path.exists(img_path):
                    print(f"Warning: Image not found: {img_path}")
                    continue
                
                img = self.preprocessor.preprocess_for_inference(img_path)
                if img is not None:
                    batch_images.append(img[0])  # Remove batch dimension
                    valid_paths.append(img_path)
            
            if len(batch_images) == 0:
                continue
            
            # Convert to numpy array
            batch_images = np.array(batch_images)
            
            # Predict
            predictions = self.model.predict(batch_images, verbose=0)
            
            # Process results
            for img_path, pred_prob in zip(valid_paths, predictions):
                pred_prob = pred_prob[0]
                pred_class = 'fake' if pred_prob >= config.CLASSIFICATION_THRESHOLD else 'real'
                confidence = pred_prob if pred_class == 'fake' else 1 - pred_prob
                
                results.append({
                    'image_path': img_path,
                    'prediction': pred_class,
                    'probability': float(pred_prob),
                    'confidence': float(confidence)
                })
            
            # Progress update
            processed = min(i + batch_size, total)
            print(f"Processed: {processed}/{total} images", end='\r')
        
        print(f"\n✓ Completed processing {len(results)} images")
        
        return results
    
    def predict_directory(self, directory_path, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
        """
        Predict on all images in a directory
        
        Args:
            directory_path (str): Path to directory containing images
            extensions (tuple): Valid image extensions
            
        Returns:
            list: List of prediction results
        """
        if not os.path.exists(directory_path):
            print(f"Error: Directory not found: {directory_path}")
            return None
        
        # Get all image files
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob(os.path.join(directory_path, f"*{ext}")))
            image_paths.extend(glob(os.path.join(directory_path, f"*{ext.upper()}")))
        
        if len(image_paths) == 0:
            print(f"No images found in: {directory_path}")
            return None
        
        print(f"\nFound {len(image_paths)} images in directory")
        
        # Predict batch
        results = self.predict_batch(image_paths)
        
        # Print summary
        if results:
            self.print_batch_summary(results)
        
        return results
    
    def predict_webcam(self, confidence_threshold=0.7, display_time=1):
        """
        Real-time deepfake detection using webcam
        
        Args:
            confidence_threshold (float): Minimum confidence to display
            display_time (int): Time to display prediction (seconds)
        """
        if self.model is None:
            print("Error: No model loaded")
            return
        
        print("\n" + "="*60)
        print("REAL-TIME DEEPFAKE DETECTION")
        print("="*60)
        print("\nStarting webcam...")
        print("Press 'q' to quit, 'c' to capture and analyze current frame")
        print("-" * 60 + "\n")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        frame_count = 0
        analyze_frame = False
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            frame_count += 1
            display_frame = frame.copy()
            
            # Analyze every 30 frames or when 'c' is pressed
            if analyze_frame or frame_count % 30 == 0:
                # Preprocess frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb_frame, config.IMG_SIZE)
                normalized = resized.astype('float32') / 255.0
                input_frame = np.expand_dims(normalized, axis=0)
                
                # Predict
                pred_prob = self.model.predict(input_frame, verbose=0)[0][0]
                pred_class = 'FAKE' if pred_prob >= config.CLASSIFICATION_THRESHOLD else 'REAL'
                confidence = pred_prob if pred_class == 'FAKE' else 1 - pred_prob
                
                # Display prediction on frame
                color = (0, 0, 255) if pred_class == 'FAKE' else (0, 255, 0)
                text = f"{pred_class}: {confidence:.2%}"
                
                cv2.putText(display_frame, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                analyze_frame = False
            
            # Display frame
            cv2.imshow('Deepfake Detection - Press Q to quit, C to capture', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                analyze_frame = True
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Webcam detection stopped")
    
    def print_prediction(self, result):
        """
        Print prediction result in a formatted way
        
        Args:
            result (dict): Prediction result dictionary
        """
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"\nImage: {os.path.basename(result['image_path'])}")
        print(f"Prediction: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Raw Probability (Fake): {result['probability']:.4f}")
        print("="*60 + "\n")
    
    def print_batch_summary(self, results):
        """
        Print summary of batch predictions
        
        Args:
            results (list): List of prediction results
        """
        print("\n" + "="*60)
        print("BATCH PREDICTION SUMMARY")
        print("="*60)
        
        total = len(results)
        real_count = sum(1 for r in results if r['prediction'] == 'real')
        fake_count = total - real_count
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        print(f"\nTotal images: {total}")
        print(f"Real: {real_count} ({real_count/total*100:.1f}%)")
        print(f"Fake: {fake_count} ({fake_count/total*100:.1f}%)")
        print(f"Average confidence: {avg_confidence:.2%}")
        
        print("\n" + "-"*60)
        print("Individual Results:")
        print("-"*60)
        
        for i, result in enumerate(results, 1):
            filename = os.path.basename(result['image_path'])
            pred = result['prediction'].upper()
            conf = result['confidence']
            print(f"{i:3d}. {filename:40s} | {pred:5s} | {conf:6.2%}")
        
        print("="*60 + "\n")
    
    def save_results_to_csv(self, results, output_path=None):
        """
        Save prediction results to CSV file
        
        Args:
            results (list): List of prediction results
            output_path (str): Path to save CSV file
        """
        import pandas as pd
        
        if output_path is None:
            output_path = os.path.join(config.RESULTS_DIR, 'inference_results.csv')
        
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        
        print(f"✓ Results saved to: {output_path}")


def main():
    """Main inference function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deepfake Detection Inference')
    parser.add_argument('--model_path', type=str, default=config.MODEL_SAVE_PATH,
                       help='Path to trained model')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--directory', type=str, help='Path to directory of images')
    parser.add_argument('--webcam', action='store_true', 
                       help='Use webcam for real-time detection')
    parser.add_argument('--save_csv', action='store_true',
                       help='Save results to CSV file')
    
    args = parser.parse_args()
    
    # Create inference engine
    inference = DeepfakeInference(model_path=args.model_path)
    
    if inference.model is None:
        print("Error: Could not load model. Exiting.")
        return
    
    # Run inference based on mode
    if args.webcam:
        inference.predict_webcam()
    
    elif args.image:
        result = inference.predict_single_image(args.image)
        if result and args.save_csv:
            inference.save_results_to_csv([result])
    
    elif args.directory:
        results = inference.predict_directory(args.directory)
        if results and args.save_csv:
            inference.save_results_to_csv(results)
    
    else:
        print("\nUsage Examples:")
        print("  Single image:  python inference.py --image path/to/image.jpg")
        print("  Directory:     python inference.py --directory path/to/images/")
        print("  Webcam:        python inference.py --webcam")
        print("  Save results:  python inference.py --directory path/ --save_csv")


if __name__ == "__main__":
    main()
