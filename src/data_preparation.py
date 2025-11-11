"""
Data Collection and Preparation Module
Handles dataset downloading, organization, and preparation for training
"""

import os
import sys
import shutil
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import json
import cv2

import config


class DownloadProgressBar(tqdm):
    """Progress bar for download operations"""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class DatasetManager:
    """
    Manages dataset downloading, extraction, and organization
    Supports popular deepfake datasets
    """
    
    DATASETS = {
        'faceforensics': {
            'name': 'FaceForensics++',
            'url': 'https://github.com/ondyari/FaceForensics',
            'description': 'Large-scale dataset with multiple manipulation methods',
            'size': '~500GB (full)',
            'type': 'manual'  # Requires manual download
        },
        'celeb-df': {
            'name': 'Celeb-DF',
            'url': 'https://github.com/yuezunli/celeb-deepfakeforensics',
            'description': 'High-quality celebrity deepfakes',
            'size': '~5.8GB',
            'type': 'manual'
        },
        'dfdc': {
            'name': 'DFDC (Deepfake Detection Challenge)',
            'url': 'https://ai.facebook.com/datasets/dfdc/',
            'description': 'Large-scale dataset from Facebook',
            'size': '~470GB',
            'type': 'manual'
        },
        'sample': {
            'name': 'Sample Dataset',
            'url': 'https://example.com/sample-deepfake-data.zip',
            'description': 'Small sample dataset for testing',
            'size': '~100MB',
            'type': 'auto'
        }
    }
    
    def __init__(self):
        """Initialize dataset manager"""
        self.base_dir = config.DATA_DIR
        self.download_dir = os.path.join(self.base_dir, 'downloads')
        self.raw_dir = os.path.join(self.base_dir, 'raw')
        
        # Create directories
        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)
    
    def list_available_datasets(self):
        """List all available datasets with information"""
        print("\n" + "="*70)
        print("AVAILABLE DEEPFAKE DATASETS")
        print("="*70 + "\n")
        
        for key, info in self.DATASETS.items():
            print(f"📦 {info['name']} ({key})")
            print(f"   Description: {info['description']}")
            print(f"   Size: {info['size']}")
            print(f"   URL: {info['url']}")
            print(f"   Download: {'Automatic' if info['type'] == 'auto' else 'Manual'}")
            print()
    
    def download_file(self, url, destination):
        """
        Download a file with progress bar
        
        Args:
            url (str): URL to download from
            destination (str): Where to save the file
        """
        print(f"\nDownloading from: {url}")
        print(f"Saving to: {destination}")
        
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                                 desc=os.path.basename(destination)) as t:
            urllib.request.urlretrieve(url, destination, reporthook=t.update_to)
        
        print("✓ Download complete!")
    
    def extract_archive(self, archive_path, extract_to):
        """
        Extract zip or tar archive
        
        Args:
            archive_path (str): Path to archive file
            extract_to (str): Directory to extract to
        """
        print(f"\nExtracting: {archive_path}")
        print(f"To: {extract_to}")
        
        os.makedirs(extract_to, exist_ok=True)
        
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith(('.tar.gz', '.tgz', '.tar')):
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"Unsupported archive format: {archive_path}")
            return False
        
        print("✓ Extraction complete!")
        return True
    
    def download_dataset(self, dataset_key):
        """
        Download a dataset if automatic download is available
        
        Args:
            dataset_key (str): Dataset identifier
        """
        if dataset_key not in self.DATASETS:
            print(f"Error: Unknown dataset '{dataset_key}'")
            self.list_available_datasets()
            return False
        
        dataset_info = self.DATASETS[dataset_key]
        
        if dataset_info['type'] == 'manual':
            print("\n" + "="*70)
            print(f"MANUAL DOWNLOAD REQUIRED: {dataset_info['name']}")
            print("="*70)
            print(f"\nPlease visit: {dataset_info['url']}")
            print(f"Size: {dataset_info['size']}")
            print(f"\nAfter downloading:")
            print(f"1. Place the dataset in: {self.raw_dir}/")
            print(f"2. Run: python src/data_preparation.py --organize --source {self.raw_dir}/your_dataset")
            return False
        
        # Automatic download (for sample datasets)
        filename = os.path.basename(dataset_info['url'])
        download_path = os.path.join(self.download_dir, filename)
        
        if os.path.exists(download_path):
            print(f"✓ Dataset already downloaded: {download_path}")
        else:
            self.download_file(dataset_info['url'], download_path)
        
        # Extract if it's an archive
        if filename.endswith(('.zip', '.tar.gz', '.tgz', '.tar')):
            extract_dir = os.path.join(self.raw_dir, dataset_key)
            self.extract_archive(download_path, extract_dir)
        
        return True
    
    def print_download_instructions(self):
        """Print detailed download instructions for major datasets"""
        print("\n" + "="*70)
        print("DATASET DOWNLOAD INSTRUCTIONS")
        print("="*70)
        
        print("\n📦 FaceForensics++")
        print("-" * 70)
        print("1. Visit: https://github.com/ondyari/FaceForensics")
        print("2. Fill out the agreement form")
        print("3. Download using provided scripts")
        print("4. Choose compression level (c0, c23, c40)")
        print("5. Recommended: c23 (good quality, manageable size)")
        
        print("\n📦 Celeb-DF")
        print("-" * 70)
        print("1. Visit: https://github.com/yuezunli/celeb-deepfakeforensics")
        print("2. Read and accept terms")
        print("3. Download via Google Drive link")
        print("4. Extract the archive")
        
        print("\n📦 DFDC (Deepfake Detection Challenge)")
        print("-" * 70)
        print("1. Visit: https://ai.facebook.com/datasets/dfdc/")
        print("2. Register and accept terms")
        print("3. Download dataset parts (multiple files)")
        print("4. Combine and extract")
        
        print("\n📦 DeepFake-TIMIT")
        print("-" * 70)
        print("1. Visit: https://www.idiap.ch/dataset/deepfaketimit")
        print("2. Complete registration form")
        print("3. Receive download link via email")
        print("4. Download and extract")
        
        print("\n" + "="*70)
        print("After downloading any dataset:")
        print("  python src/data_preparation.py --organize --source path/to/dataset")
        print("="*70 + "\n")


class DatasetOrganizer:
    """
    Organizes raw datasets into the required structure
    Handles splitting into train/val/test sets
    """
    
    def __init__(self, source_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Initialize dataset organizer
        
        Args:
            source_dir (str): Source directory containing raw images
            train_ratio (float): Training set ratio
            val_ratio (float): Validation set ratio
            test_ratio (float): Test set ratio
        """
        self.source_dir = source_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Validate ratios
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Split ratios must sum to 1.0")
    
    def find_images(self, directory, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
        """
        Recursively find all images in a directory
        
        Args:
            directory (str): Directory to search
            extensions (tuple): Valid image extensions
            
        Returns:
            list: List of image paths
        """
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(Path(directory).rglob(f'*{ext}'))
            image_paths.extend(Path(directory).rglob(f'*{ext.upper()}'))
        
        return [str(p) for p in image_paths]
    
    def detect_structure(self, source_dir):
        """
        Auto-detect dataset structure and class labels
        
        Args:
            source_dir (str): Source directory
            
        Returns:
            dict: Detected structure with classes and image counts
        """
        print("\nAnalyzing dataset structure...")
        
        structure = {
            'real': [],
            'fake': []
        }
        
        # Common naming patterns
        real_patterns = ['real', 'original', 'authentic', 'genuine', 'youtube']
        fake_patterns = ['fake', 'deepfake', 'synthetic', 'manipulated', 'forged', 
                        'deepfakes', 'faceswap', 'face2face', 'neuraltextures']
        
        # Walk through directory
        for root, dirs, files in os.walk(source_dir):
            dir_name = os.path.basename(root).lower()
            
            # Check if directory name indicates class
            is_real = any(pattern in dir_name for pattern in real_patterns)
            is_fake = any(pattern in dir_name for pattern in fake_patterns)
            
            if is_real or is_fake:
                images = self.find_images(root)
                
                if is_real:
                    structure['real'].extend(images)
                    print(f"  Found {len(images)} real images in: {root}")
                elif is_fake:
                    structure['fake'].extend(images)
                    print(f"  Found {len(images)} fake images in: {root}")
        
        print(f"\n✓ Detection complete:")
        print(f"  Real images: {len(structure['real'])}")
        print(f"  Fake images: {len(structure['fake'])}")
        
        return structure
    
    def split_dataset(self, image_paths, labels):
        """
        Split dataset into train/val/test sets with stratification
        
        Args:
            image_paths (list): List of image paths
            labels (list): Corresponding labels
            
        Returns:
            dict: Split datasets
        """
        print("\nSplitting dataset...")
        
        # First split: train and temp (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            image_paths, labels,
            test_size=(self.val_ratio + self.test_ratio),
            stratify=labels,
            random_state=config.RANDOM_SEED
        )
        
        # Second split: val and test
        val_size = self.val_ratio / (self.val_ratio + self.test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_size),
            stratify=y_temp,
            random_state=config.RANDOM_SEED
        )
        
        split_data = {
            'train': {'paths': X_train, 'labels': y_train},
            'val': {'paths': X_val, 'labels': y_val},
            'test': {'paths': X_test, 'labels': y_test}
        }
        
        print(f"✓ Split complete:")
        print(f"  Training:   {len(X_train)} images ({len(X_train)/len(image_paths)*100:.1f}%)")
        print(f"  Validation: {len(X_val)} images ({len(X_val)/len(image_paths)*100:.1f}%)")
        print(f"  Testing:    {len(X_test)} images ({len(X_test)/len(image_paths)*100:.1f}%)")
        
        return split_data
    
    def copy_images_to_structure(self, split_data, destination_base):
        """
        Copy images to organized train/val/test structure
        
        Args:
            split_data (dict): Split dataset information
            destination_base (str): Base destination directory
        """
        print("\nOrganizing images into directory structure...")
        
        for split_name, data in split_data.items():
            for img_path, label in zip(data['paths'], data['labels']):
                # Determine class name
                class_name = 'real' if label == 0 else 'fake'
                
                # Create destination path
                dest_dir = os.path.join(destination_base, split_name, class_name)
                os.makedirs(dest_dir, exist_ok=True)
                
                # Copy file
                filename = os.path.basename(img_path)
                dest_path = os.path.join(dest_dir, filename)
                
                # Handle duplicate filenames
                if os.path.exists(dest_path):
                    name, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(dest_path):
                        filename = f"{name}_{counter}{ext}"
                        dest_path = os.path.join(dest_dir, filename)
                        counter += 1
                
                shutil.copy2(img_path, dest_path)
        
        print("✓ Organization complete!")
    
    def organize_dataset(self, destination_base=None, auto_detect=True):
        """
        Main function to organize a dataset
        
        Args:
            destination_base (str): Base destination directory
            auto_detect (bool): Auto-detect dataset structure
        """
        if destination_base is None:
            destination_base = config.DATA_DIR
        
        print("\n" + "="*70)
        print("DATASET ORGANIZATION")
        print("="*70)
        print(f"\nSource: {self.source_dir}")
        print(f"Destination: {destination_base}")
        print(f"Split: Train={self.train_ratio:.0%}, Val={self.val_ratio:.0%}, Test={self.test_ratio:.0%}")
        
        # Detect structure
        if auto_detect:
            structure = self.detect_structure(self.source_dir)
        else:
            print("\nManual mode: Please organize source data into 'real' and 'fake' folders")
            structure = {
                'real': self.find_images(os.path.join(self.source_dir, 'real')),
                'fake': self.find_images(os.path.join(self.source_dir, 'fake'))
            }
        
        # Validate dataset
        if len(structure['real']) == 0 or len(structure['fake']) == 0:
            print("\n✗ Error: Could not find both real and fake images")
            print("  Please ensure your dataset contains both classes")
            return False
        
        # Check for class imbalance
        real_count = len(structure['real'])
        fake_count = len(structure['fake'])
        imbalance_ratio = max(real_count, fake_count) / min(real_count, fake_count)
        
        if imbalance_ratio > 3:
            print(f"\n⚠️  Warning: Significant class imbalance detected (ratio: {imbalance_ratio:.2f})")
            print("  Consider balancing your dataset for better results")
        
        # Combine and create labels
        all_images = structure['real'] + structure['fake']
        all_labels = [0] * len(structure['real']) + [1] * len(structure['fake'])
        
        # Split dataset
        split_data = self.split_dataset(all_images, all_labels)
        
        # Copy to organized structure
        self.copy_images_to_structure(split_data, destination_base)
        
        # Save metadata
        self.save_metadata(split_data, destination_base)
        
        print("\n" + "="*70)
        print("✓ DATASET ORGANIZATION COMPLETE")
        print("="*70)
        print(f"\nYour dataset is ready in: {destination_base}")
        print("\nNext steps:")
        print("  1. Review the organized data")
        print("  2. Train model: python main.py --train")
        
        return True
    
    def save_metadata(self, split_data, destination_base):
        """
        Save dataset metadata for reference
        
        Args:
            split_data (dict): Split dataset information
            destination_base (str): Base destination directory
        """
        metadata = {
            'source_dir': self.source_dir,
            'split_ratios': {
                'train': self.train_ratio,
                'val': self.val_ratio,
                'test': self.test_ratio
            },
            'counts': {
                'train': {
                    'real': sum(1 for l in split_data['train']['labels'] if l == 0),
                    'fake': sum(1 for l in split_data['train']['labels'] if l == 1),
                    'total': len(split_data['train']['labels'])
                },
                'val': {
                    'real': sum(1 for l in split_data['val']['labels'] if l == 0),
                    'fake': sum(1 for l in split_data['val']['labels'] if l == 1),
                    'total': len(split_data['val']['labels'])
                },
                'test': {
                    'real': sum(1 for l in split_data['test']['labels'] if l == 0),
                    'fake': sum(1 for l in split_data['test']['labels'] if l == 1),
                    'total': len(split_data['test']['labels'])
                }
            },
            'random_seed': config.RANDOM_SEED
        }
        
        metadata_path = os.path.join(destination_base, 'dataset_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"\n✓ Metadata saved to: {metadata_path}")


def main():
    """Main function for data preparation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Data Collection and Preparation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  List available datasets:
    python src/data_preparation.py --list
    
  Show download instructions:
    python src/data_preparation.py --instructions
    
  Organize dataset:
    python src/data_preparation.py --organize --source path/to/raw_dataset
    
  Organize with custom split:
    python src/data_preparation.py --organize --source path/to/raw_dataset --train 0.7 --val 0.15 --test 0.15
        """
    )
    
    parser.add_argument('--list', action='store_true',
                       help='List available datasets')
    parser.add_argument('--instructions', action='store_true',
                       help='Show download instructions')
    parser.add_argument('--download', type=str,
                       help='Download dataset (if available)')
    parser.add_argument('--organize', action='store_true',
                       help='Organize dataset into train/val/test')
    parser.add_argument('--source', type=str,
                       help='Source directory containing raw dataset')
    parser.add_argument('--destination', type=str, default=config.DATA_DIR,
                       help='Destination directory for organized data')
    parser.add_argument('--train', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--val', type=float, default=0.1,
                       help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test', type=float, default=0.1,
                       help='Test set ratio (default: 0.1)')
    
    args = parser.parse_args()
    
    # Dataset manager
    manager = DatasetManager()
    
    if args.list:
        manager.list_available_datasets()
    
    elif args.instructions:
        manager.print_download_instructions()
    
    elif args.download:
        manager.download_dataset(args.download)
    
    elif args.organize:
        if not args.source:
            print("Error: --source is required for organizing datasets")
            print("Usage: python src/data_preparation.py --organize --source path/to/dataset")
            sys.exit(1)
        
        if not os.path.exists(args.source):
            print(f"Error: Source directory not found: {args.source}")
            sys.exit(1)
        
        organizer = DatasetOrganizer(
            args.source,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test
        )
        
        organizer.organize_dataset(args.destination)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
