"""
Main Entry Point for Deepfake Detection System
Orchestrates the complete workflow: setup, training, evaluation, and inference
"""

import os
import sys
import argparse

# Add src directory to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

from src import config
from src.preprocess import ImagePreprocessor
from src.model import DeepfakeDetector
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.inference import DeepfakeInference
from src.data_preparation import DatasetManager, DatasetOrganizer


def print_banner():
    """Print application banner"""
    print("\n" + "="*70)
    print(" " * 15 + "DEEPFAKE DETECTION SYSTEM")
    print(" " * 10 + "Advanced Image Authenticity Verification")
    print("="*70 + "\n")


def setup_environment():
    """Setup the environment and create necessary directories"""
    print("Setting up environment...")
    config.create_directories()
    print("\n✓ Environment setup complete!")
    print("\nPlease organize your dataset in the following structure:")
    print("  data/train/real/    <- Real images for training")
    print("  data/train/fake/    <- Fake images for training")
    print("  data/val/real/      <- Real images for validation")
    print("  data/val/fake/      <- Fake images for validation")
    print("  data/test/real/     <- Real images for testing")
    print("  data/test/fake/     <- Fake images for testing")


def train_model(args):
    """
    Train a deepfake detection model
    
    Args:
        args: Command line arguments
    """
    print("\n" + "="*70)
    print("TRAINING MODE")
    print("="*70)
    
    # Print configuration
    config.print_config()
    
    # Create trainer
    trainer = ModelTrainer(model_type=args.model)
    
    # Train model
    print("\nStarting training...")
    history = trainer.train_with_generators(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Print summary
    trainer.print_training_summary()
    
    # Plot training history
    trainer.plot_training_history()
    
    # Fine-tune if requested and using EfficientNet
    if args.fine_tune and args.model == 'efficientnet':
        print("\n" + "="*70)
        print("FINE-TUNING MODE")
        print("="*70)
        
        trainer.fine_tune(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            epochs=args.fine_tune_epochs,
            initial_learning_rate=args.fine_tune_lr
        )
        
        trainer.print_training_summary()
        trainer.plot_training_history()
    
    print("\n✓ Training completed successfully!")
    print(f"  Model saved to: {config.MODEL_SAVE_PATH}")


def evaluate_model(args):
    """
    Evaluate a trained model
    
    Args:
        args: Command line arguments
    """
    print("\n" + "="*70)
    print("EVALUATION MODE")
    print("="*70)
    
    # Create evaluator
    evaluator = ModelEvaluator(model_path=args.model_path)
    
    if evaluator.model is None:
        print("\n✗ Error: Could not load model")
        print("  Please train a model first or check the model path")
        return
    
    # Evaluate model
    metrics = evaluator.evaluate_with_generator(
        test_dir=args.test_dir,
        batch_size=args.batch_size
    )
    
    if metrics:
        print("\n✓ Evaluation completed successfully!")
        print(f"  Results saved to: {config.RESULTS_DIR}")


def run_inference(args):
    """
    Run inference on new images
    
    Args:
        args: Command line arguments
    """
    print("\n" + "="*70)
    print("INFERENCE MODE")
    print("="*70)
    
    # Create inference engine
    inference = DeepfakeInference(model_path=args.model_path)
    
    if inference.model is None:
        print("\n✗ Error: Could not load model")
        print("  Please train a model first or check the model path")
        return
    
    # Run inference based on input type
    if args.webcam:
        print("\nStarting real-time webcam detection...")
        inference.predict_webcam()
    
    elif args.image:
        print(f"\nAnalyzing image: {args.image}")
        result = inference.predict_single_image(args.image)
        
        if result and args.save_results:
            inference.save_results_to_csv([result])
    
    elif args.directory:
        print(f"\nAnalyzing all images in: {args.directory}")
        results = inference.predict_directory(args.directory)
        
        if results and args.save_results:
            inference.save_results_to_csv(results)
    
    else:
        print("\nPlease specify input:")
        print("  --image <path>      : Analyze single image")
        print("  --directory <path>  : Analyze all images in directory")
        print("  --webcam            : Real-time webcam detection")


def prepare_data(args):
    """
    Prepare and organize dataset
    
    Args:
        args: Command line arguments
    """
    print("\n" + "="*70)
    print("DATA PREPARATION MODE")
    print("="*70)
    
    if args.list_datasets:
        manager = DatasetManager()
        manager.list_available_datasets()
    
    elif args.download_instructions:
        manager = DatasetManager()
        manager.print_download_instructions()
    
    elif args.organize_data:
        if not args.source_dir:
            print("\n✗ Error: --source_dir is required for organizing data")
            print("Usage: python main.py --prepare --organize_data --source_dir path/to/raw_data")
            return
        
        organizer = DatasetOrganizer(
            args.source_dir,
            train_ratio=args.train_split,
            val_ratio=args.val_split,
            test_ratio=args.test_split
        )
        
        organizer.organize_dataset(args.data_dir)
    
    elif args.validate_data:
        preprocessor = ImagePreprocessor()
        
        print("\nValidating Training Data...")
        if os.path.exists(config.TRAIN_DIR):
            preprocessor.validate_dataset(config.TRAIN_DIR)
        
        print("\nValidating Validation Data...")
        if os.path.exists(config.VAL_DIR):
            preprocessor.validate_dataset(config.VAL_DIR)
        
        print("\nValidating Test Data...")
        if os.path.exists(config.TEST_DIR):
            preprocessor.validate_dataset(config.TEST_DIR)
    
    elif args.extract_faces:
        if not args.source_dir or not args.output_dir:
            print("\n✗ Error: Both --source_dir and --output_dir are required for face extraction")
            print("Usage: python main.py --prepare --extract_faces --source_dir input/ --output_dir output/")
            return
        
        preprocessor = ImagePreprocessor(use_face_extraction=True)
        preprocessor.extract_and_save_faces(args.source_dir, args.output_dir)
    
    else:
        print("\nData Preparation Options:")
        print("  --list_datasets          : List available datasets")
        print("  --download_instructions  : Show download instructions")
        print("  --organize_data          : Organize raw data into train/val/test")
        print("  --validate_data          : Validate existing dataset")
        print("  --extract_faces          : Extract faces from images")


def test_system():
    """Test the system components"""
    print("\n" + "="*70)
    print("SYSTEM TEST MODE")
    print("="*70)
    
    print("\n1. Testing Configuration...")
    config.print_config()
    
    print("\n2. Testing Preprocessing...")
    preprocessor = ImagePreprocessor()
    print("✓ Preprocessor initialized")
    
    print("\n3. Testing Model Creation...")
    try:
        model = DeepfakeDetector.get_model(config.MODEL_TYPE)
        print("✓ Model created successfully")
        print(f"  Model type: {config.MODEL_TYPE}")
        print(f"  Parameters: {model.count_params():,}")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
    
    print("\n4. Testing Data Preparation...")
    try:
        manager = DatasetManager()
        print("✓ Dataset manager initialized")
    except Exception as e:
        print(f"✗ Dataset manager failed: {e}")
    
    print("\n" + "="*70)
    print("System test completed!")
    print("="*70)


def main():
    """Main function"""
    print_banner()
    
    parser = argparse.ArgumentParser(
        description='Deepfake Detection System - Complete Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Setup environment:
    python main.py --setup
    
  List available datasets:
    python main.py --prepare --list_datasets
    
  Show download instructions:
    python main.py --prepare --download_instructions
    
  Organize raw dataset:
    python main.py --prepare --organize_data --source_dir path/to/raw_data
    
  Validate existing dataset:
    python main.py --prepare --validate_data
    
  Extract faces from images:
    python main.py --prepare --extract_faces --source_dir input/ --output_dir output/
    
  Train model:
    python main.py --train --model efficientnet --epochs 50
    
  Train with fine-tuning:
    python main.py --train --model efficientnet --epochs 30 --fine_tune
    
  Evaluate model:
    python main.py --evaluate
    
  Inference on single image:
    python main.py --inference --image path/to/image.jpg
    
  Inference on directory:
    python main.py --inference --directory path/to/images/ --save_results
    
  Real-time webcam detection:
    python main.py --inference --webcam
    
  Test system:
    python main.py --test
        """
    )
    
    # Main modes
    parser.add_argument('--setup', action='store_true',
                       help='Setup environment and create directories')
    parser.add_argument('--prepare', action='store_true',
                       help='Data preparation and organization')
    parser.add_argument('--train', action='store_true',
                       help='Train a new model')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate trained model')
    parser.add_argument('--inference', action='store_true',
                       help='Run inference on new images')
    parser.add_argument('--test', action='store_true',
                       help='Test system components')
    
    # Data preparation parameters
    parser.add_argument('--list_datasets', action='store_true',
                       help='List available datasets')
    parser.add_argument('--download_instructions', action='store_true',
                       help='Show dataset download instructions')
    parser.add_argument('--organize_data', action='store_true',
                       help='Organize raw data into train/val/test structure')
    parser.add_argument('--validate_data', action='store_true',
                       help='Validate existing dataset for issues')
    parser.add_argument('--extract_faces', action='store_true',
                       help='Extract faces from images')
    parser.add_argument('--source_dir', type=str,
                       help='Source directory for data operations')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for processed data')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test_split', type=float, default=0.1,
                       help='Test set ratio (default: 0.1)')
    
    # Training parameters
    parser.add_argument('--model', type=str, default=config.MODEL_TYPE,
                       choices=['efficientnet', 'custom_cnn', 'deep_cnn'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--fine_tune', action='store_true',
                       help='Fine-tune after initial training (EfficientNet only)')
    parser.add_argument('--fine_tune_epochs', type=int, default=10,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--fine_tune_lr', type=float, default=1e-5,
                       help='Learning rate for fine-tuning')
    
    # Data directories
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR,
                       help='Base data directory')
    parser.add_argument('--train_dir', type=str, default=config.TRAIN_DIR,
                       help='Training data directory')
    parser.add_argument('--val_dir', type=str, default=config.VAL_DIR,
                       help='Validation data directory')
    parser.add_argument('--test_dir', type=str, default=config.TEST_DIR,
                       help='Test data directory')
    
    # Model path
    parser.add_argument('--model_path', type=str, default=config.MODEL_SAVE_PATH,
                       help='Path to trained model')
    
    # Inference parameters
    parser.add_argument('--image', type=str,
                       help='Path to single image for inference')
    parser.add_argument('--directory', type=str,
                       help='Path to directory of images for inference')
    parser.add_argument('--webcam', action='store_true',
                       help='Use webcam for real-time detection')
    parser.add_argument('--save_results', action='store_true',
                       help='Save inference results to CSV')
    
    args = parser.parse_args()
    
    # Execute based on mode
    try:
        if args.setup:
            setup_environment()
        
        elif args.prepare:
            prepare_data(args)
        
        elif args.train:
            train_model(args)
        
        elif args.evaluate:
            evaluate_model(args)
        
        elif args.inference:
            run_inference(args)
        
        elif args.test:
            test_system()
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
