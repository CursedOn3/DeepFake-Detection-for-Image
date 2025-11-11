"""
Image Preprocessing Module for Deepfake Detection
Handles image loading, resizing, normalization, augmentation, and face extraction
"""

import os
import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
import config

class ImagePreprocessor:
    """
    Handles all image preprocessing tasks including:
    - Loading and resizing images
    - Normalization
    - Face extraction (optional)
    - Data augmentation
    """
    
    def __init__(self, img_size=config.IMG_SIZE, use_face_extraction=config.USE_FACE_EXTRACTION):
        """
        Initialize the preprocessor
        
        Args:
            img_size (tuple): Target image size (height, width)
            use_face_extraction (bool): Whether to extract faces before processing
        """
        self.img_size = img_size
        self.use_face_extraction = use_face_extraction
        
        # Initialize face detector if needed
        if self.use_face_extraction:
            if config.FACE_DETECTION_METHOD == 'mtcnn':
                self.face_detector = MTCNN(
                    min_face_size=config.MTCNN_MIN_FACE_SIZE,
                    scale_factor=config.MTCNN_SCALE_FACTOR
                )
            elif config.FACE_DETECTION_METHOD == 'opencv_haar':
                cascade_path = cv2.data.haarcascades + config.HAAR_CASCADE_PATH
                self.face_detector = cv2.CascadeClassifier(cascade_path)
            else:
                print("Warning: Unknown face detection method. Disabling face extraction.")
                self.use_face_extraction = False
    
    def extract_face_mtcnn(self, image):
        """
        Extract face from image using MTCNN
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Cropped face image or original image if no face detected
        """
        try:
            # Convert to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            
            # Detect faces
            results = self.face_detector.detect_faces(image)
            
            if len(results) > 0:
                # Get the face with highest confidence
                best_face = max(results, key=lambda x: x['confidence'])
                
                if best_face['confidence'] >= config.FACE_CONFIDENCE_THRESHOLD:
                    x, y, width, height = best_face['box']
                    # Add padding
                    padding = 20
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    width = min(image.shape[1] - x, width + 2 * padding)
                    height = min(image.shape[0] - y, height + 2 * padding)
                    
                    face = image[y:y+height, x:x+width]
                    return face
            
            # Return original image if no face detected
            return image
        except Exception as e:
            print(f"Error in face extraction: {e}")
            return image
    
    def extract_face_opencv(self, image):
        """
        Extract face from image using OpenCV Haar Cascade
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Cropped face image or original image if no face detected
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # Get the largest face
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                face = image[y:y+h, x:x+w]
                return face
            
            return image
        except Exception as e:
            print(f"Error in face extraction: {e}")
            return image
    
    def load_and_preprocess_image(self, image_path):
        """
        Load and preprocess a single image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image array
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract face if enabled
            if self.use_face_extraction:
                if config.FACE_DETECTION_METHOD == 'mtcnn':
                    image = self.extract_face_mtcnn(image)
                elif config.FACE_DETECTION_METHOD == 'opencv_haar':
                    image = self.extract_face_opencv(image)
            
            # Resize image
            image = cv2.resize(image, self.img_size)
            
            # Normalize pixel values to [0, 1]
            image = image.astype('float32') / 255.0
            
            return image
        
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def load_dataset_from_directory(self, data_dir):
        """
        Load dataset from directory structure: data_dir/class_name/images
        
        Args:
            data_dir (str): Root directory containing class subdirectories
            
        Returns:
            tuple: (images, labels) as numpy arrays
        """
        images = []
        labels = []
        
        if not os.path.exists(data_dir):
            print(f"Warning: Directory does not exist: {data_dir}")
            return np.array([]), np.array([])
        
        # Iterate through class directories
        for class_idx, class_name in enumerate(config.CLASS_NAMES):
            class_dir = os.path.join(data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory does not exist: {class_dir}")
                continue
            
            # Get all image files
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            print(f"Loading {len(image_files)} images from {class_name}...")
            
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                img = self.load_and_preprocess_image(img_path)
                
                if img is not None:
                    images.append(img)
                    labels.append(class_idx)
        
        return np.array(images), np.array(labels)
    
    def create_data_generators(self, train_dir, val_dir, batch_size=config.BATCH_SIZE):
        """
        Create data generators for training and validation with augmentation
        
        Args:
            train_dir (str): Training data directory
            val_dir (str): Validation data directory
            batch_size (int): Batch size for generators
            
        Returns:
            tuple: (train_generator, val_generator)
        """
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=config.AUGMENTATION_CONFIG['rotation_range'],
            width_shift_range=config.AUGMENTATION_CONFIG['width_shift_range'],
            height_shift_range=config.AUGMENTATION_CONFIG['height_shift_range'],
            shear_range=config.AUGMENTATION_CONFIG['shear_range'],
            zoom_range=config.AUGMENTATION_CONFIG['zoom_range'],
            horizontal_flip=config.AUGMENTATION_CONFIG['horizontal_flip'],
            fill_mode=config.AUGMENTATION_CONFIG['fill_mode']
        )
        
        # Validation data generator (only rescaling)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary',
            classes=config.CLASS_NAMES,
            shuffle=True,
            seed=config.RANDOM_SEED
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary',
            classes=config.CLASS_NAMES,
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def preprocess_for_inference(self, image_path):
        """
        Preprocess a single image for inference
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            numpy.ndarray: Preprocessed image ready for model input (1, H, W, C)
        """
        img = self.load_and_preprocess_image(image_path)
        
        if img is None:
            return None
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img
    
    def visualize_augmentation(self, image_path, num_augmented=5):
        """
        Visualize data augmentation effects
        
        Args:
            image_path (str): Path to sample image
            num_augmented (int): Number of augmented versions to generate
        """
        import matplotlib.pyplot as plt
        
        # Load image
        img = load_img(image_path, target_size=self.img_size)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        
        # Create augmentation generator
        datagen = ImageDataGenerator(**config.AUGMENTATION_CONFIG)
        
        # Generate augmented images
        fig, axes = plt.subplots(1, num_augmented + 1, figsize=(15, 3))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Augmented images
        i = 1
        for batch in datagen.flow(x, batch_size=1):
            axes[i].imshow(batch[0].astype('uint8'))
            axes[i].set_title(f'Augmented {i}')
            axes[i].axis('off')
            i += 1
            if i > num_augmented:
                break
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, 'augmentation_samples.png'))
        plt.show()
        print(f"Augmentation samples saved to {config.RESULTS_DIR}")


    def validate_dataset(self, data_dir):
        """
        Validate dataset for common issues
        
        Args:
            data_dir (str): Directory to validate
            
        Returns:
            dict: Validation report
        """
        report = {
            'total_images': 0,
            'corrupted_images': [],
            'small_images': [],
            'class_distribution': {},
            'recommendations': []
        }
        
        print(f"\nValidating dataset: {data_dir}")
        print("-" * 60)
        
        for class_name in config.CLASS_NAMES:
            class_dir = os.path.join(data_dir, class_name)
            
            if not os.path.exists(class_dir):
                report['recommendations'].append(f"Missing class directory: {class_name}")
                continue
            
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            report['class_distribution'][class_name] = len(image_files)
            report['total_images'] += len(image_files)
            
            # Check individual images
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        report['corrupted_images'].append(img_path)
                    elif img.shape[0] < 64 or img.shape[1] < 64:
                        report['small_images'].append(img_path)
                except Exception:
                    report['corrupted_images'].append(img_path)
        
        # Generate recommendations
        if report['total_images'] < 200:
            report['recommendations'].append("Dataset is very small. Consider adding more images (minimum 500 per class recommended)")
        
        if len(report['corrupted_images']) > 0:
            report['recommendations'].append(f"Found {len(report['corrupted_images'])} corrupted images. Remove them before training.")
        
        if len(report['small_images']) > 0:
            report['recommendations'].append(f"Found {len(report['small_images'])} images smaller than 64x64. Consider removing or replacing them.")
        
        # Check class balance
        if len(report['class_distribution']) == 2:
            real_count = report['class_distribution'].get('real', 0)
            fake_count = report['class_distribution'].get('fake', 0)
            if real_count > 0 and fake_count > 0:
                imbalance_ratio = max(real_count, fake_count) / min(real_count, fake_count)
                if imbalance_ratio > 3:
                    report['recommendations'].append(f"Significant class imbalance detected (ratio: {imbalance_ratio:.2f}). Consider balancing your dataset.")
        
        # Print report
        print("\n✓ Validation complete!")
        print(f"  Total images: {report['total_images']}")
        print(f"  Class distribution: {report['class_distribution']}")
        if report['corrupted_images']:
            print(f"  Corrupted images: {len(report['corrupted_images'])}")
        if report['small_images']:
            print(f"  Small images (<64x64): {len(report['small_images'])}")
        
        if report['recommendations']:
            print("\n⚠️  Recommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        
        return report
    
    def extract_and_save_faces(self, input_dir, output_dir, min_confidence=0.9):
        """
        Extract faces from all images in a directory and save them
        
        Args:
            input_dir (str): Input directory with images
            output_dir (str): Output directory for cropped faces
            min_confidence (float): Minimum confidence for face detection
        """
        if not self.use_face_extraction:
            print("Face extraction is not enabled. Set USE_FACE_EXTRACTION=True in config.py")
            return
        
        print(f"\nExtracting faces from: {input_dir}")
        print(f"Saving to: {output_dir}")
        print(f"Method: {config.FACE_DETECTION_METHOD}")
        print("-" * 60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each class
        for class_name in config.CLASS_NAMES:
            class_input_dir = os.path.join(input_dir, class_name)
            class_output_dir = os.path.join(output_dir, class_name)
            
            if not os.path.exists(class_input_dir):
                continue
            
            os.makedirs(class_output_dir, exist_ok=True)
            
            image_files = [f for f in os.listdir(class_input_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            print(f"\nProcessing {len(image_files)} images from '{class_name}'...")
            
            successful = 0
            failed = 0
            
            for img_file in image_files:
                img_path = os.path.join(class_input_dir, img_file)
                
                try:
                    image = cv2.imread(img_path)
                    if image is None:
                        failed += 1
                        continue
                    
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Extract face
                    if config.FACE_DETECTION_METHOD == 'mtcnn':
                        face = self.extract_face_mtcnn(image_rgb)
                    else:
                        face = self.extract_face_opencv(image_rgb)
                    
                    # Save extracted face
                    output_path = os.path.join(class_output_dir, img_file)
                    face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(output_path, face_bgr)
                    
                    successful += 1
                    
                except Exception as e:
                    failed += 1
                    print(f"  Failed to process {img_file}: {e}")
            
            print(f"  ✓ Successful: {successful}")
            if failed > 0:
                print(f"  ✗ Failed: {failed}")
        
        print("\n✓ Face extraction complete!")


def test_preprocessing():
    """Test the preprocessing pipeline"""
    print("\n" + "="*60)
    print("TESTING IMAGE PREPROCESSING")
    print("="*60 + "\n")
    
    preprocessor = ImagePreprocessor()
    
    # Test loading from directories
    print("Testing data loading from train directory...")
    X_train, y_train = preprocessor.load_dataset_from_directory(config.TRAIN_DIR)
    print(f"✓ Loaded {len(X_train)} training images")
    print(f"  Shape: {X_train.shape if len(X_train) > 0 else 'No data'}")
    print(f"  Labels: {np.unique(y_train) if len(y_train) > 0 else 'No labels'}")
    
    print("\nTesting data generators...")
    try:
        train_gen, val_gen = preprocessor.create_data_generators(
            config.TRAIN_DIR, 
            config.VAL_DIR
        )
        print(f"✓ Training generator created: {train_gen.samples} samples")
        print(f"✓ Validation generator created: {val_gen.samples} samples")
    except Exception as e:
        print(f"✗ Error creating generators: {e}")
    
    # Validate dataset
    if os.path.exists(config.TRAIN_DIR):
        print("\nValidating training dataset...")
        report = preprocessor.validate_dataset(config.TRAIN_DIR)


if __name__ == "__main__":
    test_preprocessing()
