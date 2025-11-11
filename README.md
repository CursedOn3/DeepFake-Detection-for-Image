# Deepfake Detection System for Images

A comprehensive deep learning-based system for detecting deepfake images using TensorFlow/Keras. This project implements state-of-the-art CNN architectures (EfficientNetB0 and custom models) to classify images as real or fake with high accuracy.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Features

- **Multiple Model Architectures**: EfficientNetB0, Custom CNN, and Deep CNN
- **Complete Pipeline**: Data preprocessing, training, evaluation, and inference
- **Face Extraction**: Optional MTCNN or OpenCV-based face detection
- **Data Augmentation**: Comprehensive augmentation for better generalization
- **Real-time Detection**: Webcam support for live deepfake detection
- **Detailed Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualization**: Confusion matrix, ROC curves, training history plots
- **Modular Design**: Clean, maintainable, and extensible codebase
- **Production Ready**: Complete with inference scripts and model deployment

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
  - [Setup Environment](#setup-environment)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Configuration](#configuration)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster training

### Step 1: Clone the Repository

```bash
git clone https://github.com/CursedOn3/DeepFake-Detection-for-Image.git
cd DeepFake-Detection-for-Image
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install tensorflow keras opencv-python mtcnn numpy pandas scikit-learn matplotlib seaborn tqdm
```

### Step 3: Setup Environment

```bash
python main.py --setup
```

This creates the necessary directory structure for your dataset.

## 📁 Project Structure

```
DeepFake-Detection-for-Image/
│
├── src/
│   ├── config.py           # Configuration and hyperparameters
│   ├── preprocess.py       # Image preprocessing and augmentation
│   ├── model.py            # Model architectures (EfficientNet, Custom CNN)
│   ├── train.py            # Training script with callbacks
│   ├── evaluate.py         # Model evaluation and metrics
│   └── inference.py        # Inference on new images
│
├── data/                   # Dataset directory
│   ├── train/
│   │   ├── real/          # Real training images
│   │   └── fake/          # Fake training images
│   ├── val/
│   │   ├── real/          # Real validation images
│   │   └── fake/          # Fake validation images
│   └── test/
│       ├── real/          # Real test images
│       └── fake/          # Fake test images
│
├── models/                 # Saved models
│   ├── deepfake_detector.h5
│   └── checkpoint_best.h5
│
├── results/                # Evaluation results and plots
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── training_history.png
│   └── classification_report.txt
│
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── LICENSE                # License information
```

## 📊 Dataset Preparation

### Recommended Datasets

1. **FaceForensics++**: [Link](https://github.com/ondyari/FaceForensics)
2. **Celeb-DF**: [Link](https://github.com/yuezunli/celeb-deepfakeforensics)
3. **DFDC (Deepfake Detection Challenge)**: [Link](https://ai.facebook.com/datasets/dfdc/)
4. **DeepFake-TIMIT**: [Link](https://www.idiap.ch/dataset/deepfaketimit)

### Directory Structure

Organize your dataset as follows:

```
data/
├── train/
│   ├── real/    # 70-80% of real images
│   └── fake/    # 70-80% of fake images
├── val/
│   ├── real/    # 10-15% of real images
│   └── fake/    # 10-15% of fake images
└── test/
    ├── real/    # 10-15% of real images
    └── fake/    # 10-15% of fake images
```

**Recommended Split**: 80% training, 10% validation, 10% testing

### Image Requirements

- **Format**: JPG, JPEG, PNG, or BMP
- **Resolution**: Images will be automatically resized to 224×224 (configurable)
- **Quantity**: Minimum 1000 images per class recommended

## 💻 Usage

### Setup Environment

Initialize the project directories:

```bash
python main.py --setup
```

### Training

#### Basic Training

Train with default EfficientNetB0 model:

```bash
python main.py --train
```

#### Custom Training Options

```bash
# Train with custom CNN architecture
python main.py --train --model custom_cnn --epochs 50 --batch_size 32

# Train with deep CNN
python main.py --train --model deep_cnn --epochs 100

# Train EfficientNet with fine-tuning
python main.py --train --model efficientnet --epochs 30 --fine_tune
```

#### Individual Training Script

```bash
cd src
python train.py --model efficientnet --epochs 50 --batch_size 32
```

### Evaluation

Evaluate the trained model on test data:

```bash
python main.py --evaluate
```

Or specify custom model and test directory:

```bash
python main.py --evaluate --model_path models/checkpoint_best.h5 --test_dir data/test
```

Individual evaluation script:

```bash
cd src
python evaluate.py --model_path ../models/deepfake_detector.h5
```

### Inference

#### Single Image

```bash
python main.py --inference --image path/to/image.jpg
```

#### Batch Processing

```bash
python main.py --inference --directory path/to/images/ --save_results
```

#### Real-time Webcam Detection

```bash
python main.py --inference --webcam
```

Press 'C' to capture and analyze, 'Q' to quit.

#### Individual Inference Script

```bash
cd src
python inference.py --image ../data/test/fake/sample.jpg
python inference.py --directory ../data/test/fake/
python inference.py --webcam
```

### Test System

Test all components:

```bash
python main.py --test
```

## ⚙️ Configuration

Edit `src/config.py` to customize:

### Key Parameters

```python
# Model Configuration
MODEL_TYPE = 'efficientnet'  # Options: 'efficientnet', 'custom_cnn', 'deep_cnn'
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Data Augmentation
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': 0.15
}

# Face Extraction
USE_FACE_EXTRACTION = False  # Set to True to enable
FACE_DETECTION_METHOD = 'mtcnn'  # Options: 'mtcnn', 'opencv_haar'

# Classification Threshold
CLASSIFICATION_THRESHOLD = 0.5
```

## 🏗️ Model Architectures

### 1. EfficientNetB0 (Recommended)

- **Base**: Pre-trained on ImageNet
- **Transfer Learning**: Fine-tuning supported
- **Parameters**: ~4-5M trainable parameters
- **Best For**: High accuracy with reasonable training time

### 2. Custom CNN (MesoNet-inspired)

- **Architecture**: 4 conv blocks + dense layers
- **Parameters**: ~50K parameters
- **Best For**: Fast training, lightweight deployment

### 3. Deep CNN

- **Architecture**: 4 conv blocks with batch normalization
- **Parameters**: ~500K parameters
- **Best For**: Balance between accuracy and speed

## 📈 Results

### Expected Performance

With proper training on quality datasets:

| Metric     | EfficientNetB0 | Custom CNN | Deep CNN |
|-----------|----------------|------------|----------|
| Accuracy  | 94-98%         | 88-92%     | 90-95%   |
| Precision | 93-97%         | 87-91%     | 89-94%   |
| Recall    | 94-98%         | 88-92%     | 90-95%   |
| F1-Score  | 93-97%         | 87-91%     | 89-94%   |

*Results may vary based on dataset quality and size*

### Output Visualizations

Training generates the following in `results/`:

- **confusion_matrix.png**: Visual confusion matrix
- **roc_curve.png**: ROC curve with AUC score
- **training_history.png**: Loss and accuracy curves
- **classification_report.txt**: Detailed metrics per class
- **training_log.csv**: Epoch-by-epoch metrics

## 🛠️ Advanced Usage

### Custom Dataset Split

If you have a single directory with labeled data:

```python
from sklearn.model_selection import train_test_split

# Split your data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
```

### Fine-tuning Pre-trained Model

```python
from src.train import ModelTrainer

trainer = ModelTrainer('efficientnet')
trainer.load_model('models/deepfake_detector.h5')
trainer.fine_tune(epochs=10, initial_learning_rate=1e-5)
```

### Custom Model Architecture

Add your model in `src/model.py`:

```python
@staticmethod
def create_custom_model(input_shape):
    model = models.Sequential()
    # Add your layers here
    return model
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: `Import "tensorflow" could not be resolved`
- **Solution**: Install TensorFlow: `pip install tensorflow`

**Issue**: Out of memory during training
- **Solution**: Reduce batch size in `config.py`: `BATCH_SIZE = 16`

**Issue**: Webcam not working
- **Solution**: Check camera permissions and OpenCV installation

**Issue**: Low accuracy
- **Solution**: 
  - Increase dataset size
  - Enable data augmentation
  - Train for more epochs
  - Try EfficientNetB0 with fine-tuning

### GPU Support

For faster training with GPU:

```bash
pip install tensorflow-gpu==2.13.0
```

Verify GPU availability:

```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

## 📝 Development Roadmap

- [ ] Add support for video deepfake detection
- [ ] Implement additional architectures (ResNet, VGG, Xception)
- [ ] Add explainability features (Grad-CAM, LIME)
- [ ] Web interface for easy deployment
- [ ] Mobile app support
- [ ] API endpoint for cloud deployment

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow/Keras** for the deep learning framework
- **EfficientNet** paper by Tan & Le (2019)
- **MesoNet** architecture inspiration
- **FaceForensics++** dataset creators
- All open-source contributors

## 📧 Contact

For questions, issues, or suggestions:

- **GitHub Issues**: [Create an issue](https://github.com/CursedOn3/DeepFake-Detection-for-Image/issues)
- **Email**: [Your email if you want to add]

## 📚 References

1. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
2. Afchar, D., et al. (2018). MesoNet: a Compact Facial Video Forgery Detection Network
3. Rossler, A., et al. (2019). FaceForensics++: Learning to Detect Manipulated Facial Images

---

**⭐ If you find this project helpful, please consider giving it a star!**

Made with ❤️ by [CursedOn3](https://github.com/CursedOn3)