"""
Quick Setup Script for Deepfake Detection System
Automates the initial setup process
"""

import os
import sys
import subprocess

def print_banner():
    print("\n" + "="*70)
    print(" " * 15 + "DEEPFAKE DETECTION SYSTEM")
    print(" " * 20 + "Quick Setup Script")
    print("="*70 + "\n")

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python 3.8+ required. Current: {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required packages"""
    print("\nInstalling dependencies...")
    print("-" * 70)
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("\n✓ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("\n✗ Failed to install dependencies")
        print("  Please install manually: pip install -r requirements.txt")
        return False

def create_directories():
    """Create necessary directories"""
    print("\nCreating project directories...")
    
    directories = [
        'data/train/real',
        'data/train/fake',
        'data/val/real',
        'data/val/fake',
        'data/test/real',
        'data/test/fake',
        'models',
        'results',
        'results/logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}")
    
    print("\n✓ All directories created!")

def create_sample_config():
    """Display sample configuration"""
    print("\n" + "="*70)
    print("CONFIGURATION GUIDE")
    print("="*70)
    print("\nEdit src/config.py to customize:")
    print("  - Model type (efficientnet, custom_cnn, deep_cnn)")
    print("  - Image size (default: 224x224)")
    print("  - Batch size (default: 32)")
    print("  - Number of epochs (default: 50)")
    print("  - Data augmentation parameters")
    print("  - Face extraction settings")

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    
    print("\n1. Prepare Your Dataset:")
    print("   - Download a deepfake dataset (FaceForensics++, Celeb-DF, etc.)")
    print("   - Organize images into the following structure:")
    print("     data/train/real/  ← Real training images")
    print("     data/train/fake/  ← Fake training images")
    print("     data/val/real/    ← Real validation images")
    print("     data/val/fake/    ← Fake validation images")
    print("     data/test/real/   ← Real test images")
    print("     data/test/fake/   ← Fake test images")
    
    print("\n2. Configure Settings (Optional):")
    print("   - Edit src/config.py to customize parameters")
    
    print("\n3. Train Your Model:")
    print("   python main.py --train")
    print("   or")
    print("   python main.py --train --model efficientnet --epochs 50")
    
    print("\n4. Evaluate the Model:")
    print("   python main.py --evaluate")
    
    print("\n5. Run Inference:")
    print("   - Single image:  python main.py --inference --image path/to/image.jpg")
    print("   - Directory:     python main.py --inference --directory path/to/images/")
    print("   - Webcam:        python main.py --inference --webcam")
    
    print("\n6. View Results:")
    print("   - Check the 'results/' folder for visualizations and metrics")
    print("   - Trained models saved in 'models/' folder")

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Ask user for installation
    response = input("\nInstall dependencies? (y/n): ").strip().lower()
    if response == 'y':
        if not install_dependencies():
            sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Show configuration guide
    create_sample_config()
    
    # Print next steps
    print_next_steps()
    
    print("\n" + "="*70)
    print("✓ SETUP COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nFor detailed instructions, see README.md")
    print("For help: python main.py --help\n")

if __name__ == "__main__":
    main()
