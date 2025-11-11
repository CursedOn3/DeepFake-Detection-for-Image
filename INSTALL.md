# Installation Guide

Complete step-by-step installation guide for the Deepfake Detection System.

## Prerequisites

### System Requirements

**Minimum:**
- Operating System: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- RAM: 8 GB
- Storage: 5 GB free space
- Python: 3.8 or higher

**Recommended:**
- RAM: 16 GB or more
- GPU: NVIDIA GPU with CUDA support (for faster training)
- Storage: 20 GB+ (for datasets)
- Python: 3.9 or 3.10

### Software Requirements

- Python 3.8+
- pip (Python package manager)
- Git (for cloning repository)

## Installation Steps

### 1. Install Python

#### Windows
Download from [python.org](https://www.python.org/downloads/)
- ✅ Check "Add Python to PATH" during installation
- Verify: `python --version`

#### macOS
```bash
brew install python@3.10
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.10 python3-pip
```

### 2. Clone the Repository

```bash
git clone https://github.com/CursedOn3/DeepFake-Detection-for-Image.git
cd DeepFake-Detection-for-Image
```

### 3. Create Virtual Environment (Recommended)

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

#### Option A: Automatic Installation
```bash
python setup.py
```

#### Option B: Manual Installation
```bash
pip install -r requirements.txt
```

#### Option C: Individual Packages
```bash
pip install tensorflow>=2.13.0
pip install keras>=2.13.0
pip install opencv-python>=4.8.0
pip install mtcnn>=0.1.1
pip install numpy>=1.24.0
pip install pandas>=2.0.0
pip install scikit-learn>=1.3.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install Pillow>=10.0.0
pip install tqdm>=4.65.0
```

### 5. Verify Installation

```bash
python main.py --test
```

Expected output:
```
✓ Configuration loaded
✓ Preprocessor initialized
✓ Model created successfully
```

## GPU Support (Optional but Recommended)

### Install CUDA (NVIDIA GPUs only)

#### 1. Check GPU Compatibility
```bash
nvidia-smi
```

#### 2. Install CUDA Toolkit
Download from [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
- Recommended: CUDA 11.8 or 12.0

#### 3. Install cuDNN
Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)

#### 4. Install TensorFlow GPU
```bash
pip install tensorflow-gpu==2.13.0
```

#### 5. Verify GPU Detection
```python
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

Expected output:
```
GPU Available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Platform-Specific Instructions

### Windows

1. **Install Visual C++ Redistributable**
   - Download from [Microsoft](https://aka.ms/vs/17/release/vc_redist.x64.exe)
   - Required for TensorFlow

2. **Path Configuration**
   ```bash
   # Add to PATH if needed
   set PATH=%PATH%;C:\Python310;C:\Python310\Scripts
   ```

3. **PowerShell Execution Policy**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

### macOS

1. **Install Xcode Command Line Tools**
   ```bash
   xcode-select --install
   ```

2. **Install Homebrew** (if not installed)
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. **Apple Silicon (M1/M2) Specific**
   ```bash
   # Install Rosetta if needed
   softwareupdate --install-rosetta
   
   # Use tensorflow-macos for better performance
   pip install tensorflow-macos
   pip install tensorflow-metal  # GPU acceleration
   ```

### Linux (Ubuntu)

1. **Install System Dependencies**
   ```bash
   sudo apt update
   sudo apt install -y python3-dev python3-pip
   sudo apt install -y libsm6 libxext6 libxrender-dev
   sudo apt install -y libgl1-mesa-glx  # For OpenCV
   ```

2. **Install for GPU (NVIDIA)**
   ```bash
   # Install NVIDIA drivers
   sudo ubuntu-drivers autoinstall
   
   # Install CUDA
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
   sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
   sudo apt update
   sudo apt install cuda
   ```

## Docker Installation (Alternative)

### Create Dockerfile

```dockerfile
FROM tensorflow/tensorflow:2.13.0-gpu

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py", "--help"]
```

### Build and Run

```bash
# Build image
docker build -t deepfake-detector .

# Run container
docker run -it --gpus all -v $(pwd)/data:/app/data deepfake-detector

# Train model
docker run -it --gpus all -v $(pwd)/data:/app/data deepfake-detector python main.py --train
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: `pip: command not found`
**Solution:**
```bash
# Windows
python -m ensurepip --upgrade

# macOS/Linux
sudo apt install python3-pip
```

#### Issue: `tensorflow` import error
**Solution:**
```bash
pip uninstall tensorflow tensorflow-gpu
pip install tensorflow==2.13.0
```

#### Issue: `cv2` import error
**Solution:**
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python-headless==4.8.0.74
```

#### Issue: CUDA not found
**Solution:**
1. Verify CUDA installation: `nvcc --version`
2. Add to PATH:
   - Windows: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`
   - Linux: `export PATH=/usr/local/cuda/bin:$PATH`
3. Reinstall tensorflow-gpu

#### Issue: Memory error during training
**Solution:**
```python
# Edit src/config.py
BATCH_SIZE = 8  # Reduce from 32
```

#### Issue: Slow training without GPU
**Solution:**
1. Reduce image size in config.py
2. Use custom_cnn model (lighter)
3. Consider cloud GPU (Google Colab, AWS, etc.)

### Package Version Conflicts

If you encounter version conflicts:

```bash
# Create fresh environment
python -m venv fresh_env
source fresh_env/bin/activate  # or fresh_env\Scripts\activate on Windows

# Install specific versions
pip install tensorflow==2.13.0 numpy==1.24.0 keras==2.13.0
pip install -r requirements.txt
```

## Verification Checklist

After installation, verify:

```bash
# 1. Python version
python --version  # Should be 3.8+

# 2. TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"  # Should be 2.13.0+

# 3. GPU (if applicable)
python -c "import tensorflow as tf; print('GPU:', len(tf.config.list_physical_devices('GPU')) > 0)"

# 4. Other packages
python -c "import cv2, mtcnn, sklearn, matplotlib; print('All packages OK')"

# 5. Project structure
python main.py --test
```

## Update Installation

To update the project:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install --upgrade -r requirements.txt
```

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove project
cd ..
rm -rf DeepFake-Detection-for-Image

# Remove virtual environment (if created outside project)
rm -rf venv
```

## Cloud Installation

### Google Colab

```python
# Clone repository
!git clone https://github.com/CursedOn3/DeepFake-Detection-for-Image.git
%cd DeepFake-Detection-for-Image

# Install dependencies
!pip install -r requirements.txt

# Mount Google Drive for data
from google.colab import drive
drive.mount('/content/drive')

# Train model
!python main.py --train --epochs 20
```

### AWS EC2

```bash
# Launch Deep Learning AMI
# Connect via SSH
ssh -i your-key.pem ubuntu@ec2-instance-ip

# Clone and setup
git clone https://github.com/CursedOn3/DeepFake-Detection-for-Image.git
cd DeepFake-Detection-for-Image
pip install -r requirements.txt

# Train with GPU
python main.py --train --model efficientnet --epochs 50
```

## Support

If you encounter issues not covered here:

1. **Check existing issues**: [GitHub Issues](https://github.com/CursedOn3/DeepFake-Detection-for-Image/issues)
2. **Create new issue**: Provide error message, OS, Python version
3. **Consult documentation**: README.md, QUICKSTART.md

## Next Steps

After successful installation:

1. ✅ Run `python main.py --setup` to create directories
2. ✅ Prepare your dataset (see QUICKSTART.md)
3. ✅ Review configuration (src/config.py)
4. ✅ Start training!

---

**Installation successful? Start with [QUICKSTART.md](QUICKSTART.md)**
