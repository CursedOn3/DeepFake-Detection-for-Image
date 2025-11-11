# Data Collection and Preparation Guide

Complete guide for collecting, organizing, and preparing deepfake image datasets for training.

## 📊 Overview

This guide covers:
1. Finding and downloading datasets
2. Organizing raw data into proper structure
3. Splitting data (train/validation/test)
4. Validating dataset quality
5. Face extraction (optional)
6. Data augmentation strategies

## 🗂️ Dataset Structure

The system expects the following directory structure:

```
data/
├── train/
│   ├── real/    # Real training images
│   └── fake/    # Fake training images
├── val/
│   ├── real/    # Real validation images
│   └── fake/    # Fake validation images
└── test/
    ├── real/    # Real test images
    └── fake/    # Fake test images
```

## 📥 Step 1: Dataset Sources

### Recommended Datasets

#### 1. FaceForensics++ ⭐ **Most Popular**

**Description**: Large-scale video dataset with various manipulation methods
- **Size**: ~500GB (full), ~38GB (c23 compressed)
- **Content**: 1000+ videos, multiple forgery methods
- **Quality**: High quality, professionally created
- **Manipulation Types**: DeepFakes, Face2Face, FaceSwap, NeuralTextures

**How to Download**:
```bash
# Visit and fill form
https://github.com/ondyari/FaceForensics

# After approval, use their download script
python download-FaceForensics.py

# Recommended: Download c23 (compressed, good quality)
python download-FaceForensics.py -d all -c c23 -t videos
```

#### 2. Celeb-DF (v2)

**Description**: Celebrity deepfake dataset
- **Size**: ~5.8GB
- **Content**: 590 real videos, 5,639 deepfake videos
- **Quality**: High visual quality deepfakes
- **Notable**: More challenging than FaceForensics++

**How to Download**:
```bash
# Visit and accept terms
https://github.com/yuezunli/celeb-deepfakeforensics

# Download from provided Google Drive link
# Extract the archive:
unzip Celeb-DF-v2.zip
```

#### 3. DFDC (Deepfake Detection Challenge)

**Description**: Facebook's large-scale dataset
- **Size**: ~470GB
- **Content**: 100,000+ videos
- **Quality**: Diverse, crowdsourced content
- **Notable**: Most diverse dataset

**How to Download**:
```bash
# Register at
https://ai.facebook.com/datasets/dfdc/

# Use their download tool
dfdc download --all
```

#### 4. DeepFake-TIMIT

**Description**: Audio-visual deepfake dataset
- **Size**: ~15GB
- **Content**: 320 videos (real + fake)
- **Quality**: Controlled environment
- **Notable**: Good for initial testing

**How to Download**:
```bash
# Fill registration form
https://www.idiap.ch/dataset/deepfaketimit

# Download via provided link after approval
```

### Free Sample Datasets

For testing and learning:

#### DeeperForensics-1.0
- **Size**: ~10GB (mini version)
- **Link**: https://github.com/EndlessSora/DeeperForensics-1.0
- **Good for**: Initial experiments

#### Kaggle Deepfake Detection
- **Size**: Varies
- **Link**: https://www.kaggle.com/c/deepfake-detection-challenge
- **Good for**: Competitions and benchmarks

## 🔧 Step 2: Organize Raw Data

### Automatic Organization

Use the built-in data preparation tool:

```bash
# List available datasets
python main.py --prepare --list_datasets

# Show detailed download instructions
python main.py --prepare --download_instructions

# Organize raw data (auto-detect structure)
python main.py --prepare --organize_data --source_dir path/to/raw_data

# Custom split ratios
python main.py --prepare --organize_data \
    --source_dir path/to/raw_data \
    --train_split 0.7 \
    --val_split 0.15 \
    --test_split 0.15
```

### Manual Organization

If you have a mixed dataset:

1. **Create source structure**:
```
raw_data/
├── real/       # All real images
└── fake/       # All fake images
```

2. **Run organizer**:
```bash
python main.py --prepare --organize_data --source_dir raw_data/
```

The tool will:
- ✅ Auto-detect image files
- ✅ Split into train/val/test (80/10/10 by default)
- ✅ Maintain class balance
- ✅ Create proper directory structure
- ✅ Save metadata for reference

### Expected Output

```
data/
├── train/
│   ├── real/    # 80% of real images
│   └── fake/    # 80% of fake images
├── val/
│   ├── real/    # 10% of real images
│   └── fake/    # 10% of fake images
├── test/
│   ├── real/    # 10% of real images
│   └── fake/    # 10% of fake images
└── dataset_metadata.json
```

## ✅ Step 3: Validate Dataset

Validate your dataset for common issues:

```bash
python main.py --prepare --validate_data
```

This checks for:
- ✅ Corrupted images
- ✅ Images too small (<64x64)
- ✅ Class imbalance
- ✅ Minimum dataset size
- ✅ File format issues

### Validation Report Example

```
Validating dataset: data/train
────────────────────────────────────────
✓ Validation complete!
  Total images: 2000
  Class distribution: {'real': 1000, 'fake': 1000}
  Corrupted images: 0
  Small images (<64x64): 0

⚠️  Recommendations:
  - Dataset is ready for training
```

## 🎭 Step 4: Face Extraction (Optional)

Extract and crop faces before training for better results:

### Enable Face Extraction

Edit `src/config.py`:
```python
# Enable face extraction
USE_FACE_EXTRACTION = True

# Choose method: 'mtcnn' or 'opencv_haar'
FACE_DETECTION_METHOD = 'mtcnn'

# Set confidence threshold
FACE_CONFIDENCE_THRESHOLD = 0.9
```

### Extract Faces

```bash
# Extract faces from organized dataset
python main.py --prepare --extract_faces \
    --source_dir data/train \
    --output_dir data/train_faces
```

### When to Use Face Extraction

**Use when**:
- ✅ Images contain full scenes (not just faces)
- ✅ Multiple people in images
- ✅ You want to focus on facial features
- ✅ Reducing background noise

**Skip when**:
- ❌ Dataset already contains cropped faces
- ❌ Face detection is unreliable for your data
- ❌ You want to preserve context

## 📏 Step 5: Image Requirements

### Size Requirements

**Minimum**: 64x64 pixels  
**Recommended**: 224x224 pixels (for EfficientNet)  
**Maximum**: Any size (will be resized)

Images are automatically resized during preprocessing.

### Format Requirements

**Supported**: JPG, JPEG, PNG, BMP  
**Color**: RGB (3 channels)  
**Bit depth**: 8-bit per channel

### Quality Requirements

- No extreme compression artifacts
- Clear facial features
- Adequate lighting
- Minimal blur

## 📊 Step 6: Dataset Size Guidelines

### Minimum (For Testing)
- **Per class**: 100 images
- **Total**: 200 images
- **Expected accuracy**: 70-80%

### Recommended (For Production)
- **Per class**: 1,000+ images
- **Total**: 2,000+ images
- **Expected accuracy**: 90-95%

### Optimal (For Best Results)
- **Per class**: 10,000+ images
- **Total**: 20,000+ images
- **Expected accuracy**: 95-98%

## ⚖️ Step 7: Class Balance

### Balanced Dataset (Recommended)
```
Real: 1000 images
Fake: 1000 images
Ratio: 1:1 ✓
```

### Acceptable Imbalance
```
Real: 1500 images
Fake: 1000 images
Ratio: 1.5:1 ✓
```

### Problematic Imbalance
```
Real: 3000 images
Fake: 500 images
Ratio: 6:1 ✗
```

**Solutions for Imbalance**:
1. Collect more data for minority class
2. Use data augmentation heavily
3. Apply class weights during training
4. Use SMOTE or similar techniques

## 🎨 Step 8: Data Augmentation

Already configured in `src/config.py`:

```python
AUGMENTATION_CONFIG = {
    'rotation_range': 20,        # ±20 degrees rotation
    'width_shift_range': 0.2,    # Horizontal shift
    'height_shift_range': 0.2,   # Vertical shift
    'shear_range': 0.15,         # Shear transformation
    'zoom_range': 0.15,          # Zoom in/out
    'horizontal_flip': True,     # Mirror flip
    'fill_mode': 'nearest'       # Fill mode for transformations
}
```

These are applied automatically during training!

## 📋 Complete Workflow Example

### Starting from FaceForensics++

```bash
# 1. Download FaceForensics++ (after getting approval)
python download-FaceForensics.py -d all -c c23

# 2. Extract frames from videos (use their tools or ffmpeg)
for video in videos/*.mp4; do
    ffmpeg -i "$video" -vf fps=1 frames/"$(basename "$video" .mp4)"_%04d.jpg
done

# 3. Organize into real/fake folders
mkdir -p raw_data/real raw_data/fake
mv frames/original_* raw_data/real/
mv frames/manipulated_* raw_data/fake/

# 4. Organize into train/val/test
python main.py --prepare --organize_data --source_dir raw_data/

# 5. Validate dataset
python main.py --prepare --validate_data

# 6. Optional: Extract faces
python main.py --prepare --extract_faces \
    --source_dir data/train \
    --output_dir data/train_faces

# 7. Start training!
python main.py --train --model efficientnet --epochs 50
```

## 🔍 Data Quality Checklist

Before training, verify:

- [ ] Dataset is organized in correct structure
- [ ] Both classes (real/fake) are present
- [ ] Minimum 200 images per class
- [ ] No corrupted images
- [ ] Class balance ratio < 3:1
- [ ] Images are adequate quality
- [ ] Train/val/test split is correct (80/10/10)
- [ ] No data leakage between splits
- [ ] Metadata file is generated

## 🐛 Common Issues and Solutions

### Issue: "Could not find both real and fake images"
**Solution**: 
- Ensure raw data has folders named 'real' and 'fake'
- Or use auto-detection by having keywords in folder names

### Issue: "Significant class imbalance detected"
**Solution**:
- Collect more data for minority class
- Use heavy data augmentation
- Apply class weights in training

### Issue: "Dataset is very small"
**Solution**:
- Collect more data (aim for 500+ per class)
- Use transfer learning (EfficientNet)
- Enable all augmentation options

### Issue: "Face extraction fails"
**Solution**:
- Try different detection method (MTCNN vs OpenCV)
- Lower confidence threshold
- Ensure images contain visible faces

### Issue: "Out of disk space"
**Solution**:
- Use compressed dataset versions (c23 vs c0)
- Delete raw data after organization
- Use external storage

## 📊 Dataset Statistics

After organization, check `data/dataset_metadata.json`:

```json
{
    "source_dir": "raw_data/",
    "split_ratios": {
        "train": 0.8,
        "val": 0.1,
        "test": 0.1
    },
    "counts": {
        "train": {
            "real": 800,
            "fake": 800,
            "total": 1600
        },
        "val": {
            "real": 100,
            "fake": 100,
            "total": 200
        },
        "test": {
            "real": 100,
            "fake": 100,
            "total": 200
        }
    },
    "random_seed": 42
}
```

## 🚀 Quick Start Commands

```bash
# Complete workflow in 3 commands:

# 1. Setup
python main.py --setup

# 2. Organize your raw data
python main.py --prepare --organize_data --source_dir path/to/raw_data

# 3. Train
python main.py --train
```

## 📚 Additional Resources

### Dataset Papers
- FaceForensics++: [arXiv:1901.08971](https://arxiv.org/abs/1901.08971)
- Celeb-DF: [arXiv:1909.12962](https://arxiv.org/abs/1909.12962)
- DFDC: [arXiv:2006.07397](https://arxiv.org/abs/2006.07397)

### Tools
- **FFmpeg**: Extract frames from videos
- **ImageMagick**: Batch image processing
- **Python-PIL**: Image manipulation
- **OpenCV**: Computer vision operations

### Best Practices
1. Always keep a copy of raw data
2. Document your preprocessing steps
3. Use version control for datasets
4. Maintain consistent naming conventions
5. Regularly validate data quality

---

**Need Help?** 
- Check `README.md` for general documentation
- See `QUICKSTART.md` for quick setup
- Review `main.py --help` for all commands

**Ready to train?** After preparing your data:
```bash
python main.py --train --model efficientnet --epochs 50
```
