"""
Create a minimal test dataset for verifying the project works
"""
import os
import numpy as np
from PIL import Image
from pathlib import Path

def create_dummy_dataset(num_train=20, num_val=5, num_test=5):
    """
    Create a small dummy dataset for testing
    
    Args:
        num_train: Number of images per class for training
        num_val: Number of images per class for validation
        num_test: Number of images per class for testing
    """
    
    splits = {
        'train': num_train,
        'val': num_val,
        'test': num_test
    }
    labels = ['real', 'fake']
    
    print("\n" + "="*60)
    print("Creating Test Dataset")
    print("="*60)
    
    total_images = 0
    
    for split, num_images in splits.items():
        for label in labels:
            dir_path = Path(f'data/{split}/{label}')
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create dummy images (224x224, RGB)
            for i in range(num_images):
                # Create different patterns for real vs fake
                if label == 'real':
                    # More natural colors for "real"
                    # Simulate face-like colors (skin tones)
                    r = np.random.randint(150, 220, (224, 224), dtype=np.uint8)
                    g = np.random.randint(120, 180, (224, 224), dtype=np.uint8)
                    b = np.random.randint(100, 160, (224, 224), dtype=np.uint8)
                    img_array = np.stack([r, g, b], axis=2)
                else:
                    # Slightly different pattern for "fake"
                    # More artificial/random colors
                    r = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
                    g = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
                    b = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
                    img_array = np.stack([r, g, b], axis=2)
                
                img = Image.fromarray(img_array)
                img.save(f'{dir_path}/image_{i:03d}.jpg', quality=95)
                total_images += 1
            
            print(f"✓ Created {num_images} images in {split}/{label}/")
    
    print("\n" + "="*60)
    print("✅ Test Dataset Created Successfully!")
    print("="*60)
    print(f"\nTotal images created: {total_images}")
    print(f"\nDataset structure:")
    print(f"  train/real: {num_train} images")
    print(f"  train/fake: {num_train} images")
    print(f"  val/real: {num_val} images")
    print(f"  val/fake: {num_val} images")
    print(f"  test/real: {num_test} images")
    print(f"  test/fake: {num_test} images")
    print("\n✓ Ready for training!")

if __name__ == "__main__":
    create_dummy_dataset(num_train=20, num_val=5, num_test=5)
