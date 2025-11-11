# Quick Start Guide

This guide will help you get started with the Deepfake Detection System in minutes.

## 🚀 Quick Setup (5 minutes)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or run the automated setup:

```bash
python setup.py
```

### Step 2: Prepare Sample Data

Create a minimal dataset for testing:

```bash
# Create directories
python main.py --setup

# Add at least 100 images to each:
# - data/train/real/
# - data/train/fake/
# - data/val/real/
# - data/val/fake/
# - data/test/real/
# - data/test/fake/
```

### Step 3: Test the System

```bash
python main.py --test
```

## 🎯 Training Your First Model (10 minutes)

### Quick Training (Small Dataset)

```bash
# Train with custom CNN (faster)
python main.py --train --model custom_cnn --epochs 10 --batch_size 16
```

### Full Training (Large Dataset)

```bash
# Train with EfficientNetB0 (better accuracy)
python main.py --train --model efficientnet --epochs 50 --batch_size 32
```

## 📊 Evaluate Your Model

```bash
python main.py --evaluate
```

Results will be saved in the `results/` folder:
- Confusion matrix
- ROC curve
- Classification report
- Training history plots

## 🔍 Run Inference

### Single Image

```bash
python main.py --inference --image path/to/test_image.jpg
```

### Batch Processing

```bash
python main.py --inference --directory path/to/images/ --save_results
```

### Real-time Webcam

```bash
python main.py --inference --webcam
```

Press 'C' to capture and analyze, 'Q' to quit.

## ⚙️ Quick Configuration

Edit `src/config.py` for basic settings:

```python
# Model choice
MODEL_TYPE = 'efficientnet'  # or 'custom_cnn', 'deep_cnn'

# Image size
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Training
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
```

## 📚 Common Commands

### Training

```bash
# Basic training
python main.py --train

# Custom parameters
python main.py --train --model efficientnet --epochs 30 --batch_size 16

# With fine-tuning
python main.py --train --model efficientnet --epochs 20 --fine_tune
```

### Evaluation

```bash
# Default evaluation
python main.py --evaluate

# Custom model
python main.py --evaluate --model_path models/checkpoint_best.h5
```

### Inference

```bash
# Single image
python main.py --inference --image sample.jpg

# Directory of images
python main.py --inference --directory test_images/

# Save results to CSV
python main.py --inference --directory test_images/ --save_results

# Real-time detection
python main.py --inference --webcam
```

## 🐛 Troubleshooting

### Issue: Module not found
```bash
pip install -r requirements.txt
```

### Issue: Out of memory
Edit `src/config.py`:
```python
BATCH_SIZE = 8  # Reduce from 32
```

### Issue: No training data found
Make sure images are in the correct directory structure:
```
data/
├── train/
│   ├── real/  ← Add images here
│   └── fake/  ← Add images here
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

### Issue: Low accuracy
- Increase dataset size (minimum 500 images per class)
- Train for more epochs (50-100)
- Use EfficientNetB0 model
- Enable fine-tuning

## 📈 Expected Timeline

| Task | Time | Command |
|------|------|---------|
| Setup | 5 min | `python setup.py` |
| Prepare data | 10-30 min | Manual organization |
| Quick training | 10-20 min | `--model custom_cnn --epochs 10` |
| Full training | 1-3 hours | `--model efficientnet --epochs 50` |
| Evaluation | 2-5 min | `python main.py --evaluate` |
| Inference | < 1 sec/image | `python main.py --inference --image` |

## 🎓 Learning Path

### Beginner
1. Run `python main.py --test` to verify setup
2. Start with custom_cnn model (faster training)
3. Use a small dataset (200 images per class)
4. Train for 10 epochs

### Intermediate
1. Use EfficientNetB0 model
2. Prepare 1000+ images per class
3. Train for 30-50 epochs
4. Enable data augmentation
5. Experiment with hyperparameters

### Advanced
1. Implement custom architectures
2. Use large datasets (10,000+ images)
3. Fine-tune pre-trained models
4. Add face extraction preprocessing
5. Deploy as API or web service

## 📞 Need Help?

- **Full Documentation**: See `README.md`
- **Configuration Help**: Check `src/config.py`
- **Issues**: Create a GitHub issue
- **Command Help**: `python main.py --help`

## ✅ Checklist

Before training:
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset organized in correct structure
- [ ] Minimum 100 images per class (500+ recommended)
- [ ] Training/validation/test split prepared
- [ ] Configuration reviewed (`src/config.py`)

After training:
- [ ] Model saved in `models/` folder
- [ ] Training history plot generated
- [ ] Validation accuracy > 80%
- [ ] Model evaluated on test set
- [ ] Results saved in `results/` folder

Ready for deployment:
- [ ] Model tested on new images
- [ ] Inference speed acceptable
- [ ] Prediction accuracy verified
- [ ] Documentation updated

## 🎉 Success Indicators

Your model is ready when:
- ✅ Test accuracy > 90%
- ✅ Precision and recall > 85%
- ✅ ROC-AUC > 0.95
- ✅ Inference time < 1 second per image
- ✅ Predictions are consistent

---

**Happy Detecting! 🔍**

For more details, see the full [README.md](README.md)
