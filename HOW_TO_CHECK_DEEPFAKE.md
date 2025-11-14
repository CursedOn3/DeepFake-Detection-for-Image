# 🔍 How to Check if a Photo is a Deepfake

This guide shows you how to check if any photo is a deepfake using this detection system.

---

## 🚀 Quick Start (3 Simple Steps)

### Step 1: Make sure you have a trained model
```bash
# If you haven't trained a model yet, use the provided one
# OR train a quick test model:
python main.py --train --model custom_cnn --epochs 2
```

### Step 2: Place your photo anywhere accessible

### Step 3: Check the photo
```bash
python check_deepfake.py path/to/your/photo.jpg
```

**That's it!** You'll get an instant result showing if it's REAL or FAKE.

---

## 📋 Method 1: Simple Script (Recommended for Users)

### Check a Single Photo
```bash
python check_deepfake.py photo.jpg
```

**Output Example:**
```
======================================================================
                    DEEPFAKE DETECTION RESULT
======================================================================

📁 Image: suspicious_photo.jpg
📍 Path: C:/Users/You/Downloads/suspicious_photo.jpg

----------------------------------------------------------------------

⚠️  Prediction: FAKE (Deepfake)
📊 Confidence: 87.34%
🔢 Raw Score: 0.8734 (>0.5 = Fake, <0.5 = Real)

----------------------------------------------------------------------

💡 Interpretation Guide:
   → High Confidence - Result is reliable

======================================================================
```

### Check Multiple Photos
```bash
python check_deepfake.py photo1.jpg photo2.jpg photo3.jpg --batch
```

**Output Example:**
```
======================================================================
                        BATCH SUMMARY
======================================================================

📊 Total Images Processed: 3/3
✅ Real (Authentic): 2 (66.7%)
⚠️  Fake (Deepfake): 1 (33.3%)

----------------------------------------------------------------------
Detailed Results:
----------------------------------------------------------------------
1. photo1.jpg: ✅ REAL (Authentic) (92.15%)
2. photo2.jpg: ⚠️ FAKE (Deepfake) (87.34%)
3. photo3.jpg: ✅ REAL (Authentic) (78.52%)
======================================================================
```

### Use a Custom Model
```bash
python check_deepfake.py photo.jpg --model models/my_custom_model.h5
```

---

## 📋 Method 2: Main Script (More Options)

### Check a Single Image
```bash
python main.py --inference --image path/to/photo.jpg
```

### Check All Images in a Directory
```bash
python main.py --inference --directory path/to/photos/ --save_results
```
This will:
- Check all images in the directory
- Save results to `results/inference_results.csv`

### Real-Time Webcam Detection
```bash
python main.py --inference --webcam
```
Press 'q' to quit, 's' to save screenshot

---

## 🎯 Understanding the Results

### Confidence Levels

| Confidence | Meaning | Action |
|------------|---------|--------|
| **90-100%** | Very High Confidence | Result is highly reliable |
| **75-90%** | High Confidence | Result is reliable |
| **60-75%** | Moderate Confidence | Result is fairly reliable |
| **50-60%** | Low Confidence | Uncertain - manual review recommended |

### Raw Score Interpretation

The model outputs a score between 0 and 1:
- **0.0 - 0.5**: Likely REAL (authentic)
- **0.5 - 1.0**: Likely FAKE (deepfake)
- **Close to 0.5**: Uncertain, needs human review

---

## 💻 Using in Python Code

### Example 1: Check Single Image
```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('models/deepfake_detector.h5')

# Load and preprocess image
img = Image.open('photo.jpg').convert('RGB')
img = img.resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)[0][0]

if prediction > 0.5:
    print(f"FAKE - Confidence: {prediction*100:.2f}%")
else:
    print(f"REAL - Confidence: {(1-prediction)*100:.2f}%")
```

### Example 2: Check Multiple Images
```python
import os
from check_deepfake import load_model, predict_deepfake

# Load model once
model = load_model()

# Check multiple images
image_paths = ['photo1.jpg', 'photo2.jpg', 'photo3.jpg']

for image_path in image_paths:
    result = predict_deepfake(model, image_path)
    print(f"{image_path}: {result['label']} ({result['confidence']:.2f}%)")
```

### Example 3: Integrate into Your App
```python
from check_deepfake import load_model, predict_deepfake

class DeepfakeChecker:
    def __init__(self):
        self.model = load_model()
    
    def check_image(self, image_path):
        result = predict_deepfake(self.model, image_path)
        return {
            'is_fake': result['is_fake'],
            'confidence': result['confidence'],
            'label': result['label']
        }

# Usage
checker = DeepfakeChecker()
result = checker.check_image('uploaded_photo.jpg')

if result['is_fake']:
    print("⚠️ Warning: This appears to be a deepfake!")
else:
    print("✅ This appears to be authentic")
```

---

## 🌐 Creating a Web Interface

### Simple Flask App Example

```python
from flask import Flask, request, jsonify, render_template
from check_deepfake import load_model, predict_deepfake
import os

app = Flask(__name__)
model = load_model()

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head><title>Deepfake Detector</title></head>
    <body>
        <h1>🔍 Deepfake Detection</h1>
        <form method="POST" action="/check" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <button type="submit">Check Image</button>
        </form>
    </body>
    </html>
    '''

@app.route('/check', methods=['POST'])
def check():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    # Check if deepfake
    result = predict_deepfake(model, filepath)
    
    # Clean up
    os.remove(filepath)
    
    return jsonify({
        'filename': file.filename,
        'prediction': result['label'],
        'is_fake': result['is_fake'],
        'confidence': f"{result['confidence']:.2f}%"
    })

if __name__ == '__main__':
    app.run(debug=True)
```

Save as `web_app.py` and run:
```bash
pip install flask
python web_app.py
```

Then open: http://localhost:5000

---

## 📱 Common Use Cases

### 1. Social Media Verification
Check photos before sharing:
```bash
python check_deepfake.py downloaded_image.jpg
```

### 2. Batch Processing
Check an entire folder:
```bash
# Get all jpg files
python check_deepfake.py folder/*.jpg --batch
```

### 3. Automated Pipeline
```bash
# Watch folder and auto-check new images
# (Requires custom script with file watching)
```

### 4. API Integration
```python
# Use as part of your content moderation system
from check_deepfake import load_model, predict_deepfake

def moderate_image(image_path):
    result = predict_deepfake(model, image_path)
    if result['is_fake'] and result['confidence'] > 80:
        return "REJECT - Likely deepfake"
    return "APPROVE"
```

---

## ⚠️ Important Notes

### Accuracy Considerations

1. **Model Quality**: Accuracy depends on training data quality
   - Trained on synthetic data: ~50-60% accuracy
   - Trained on real deepfake datasets: **85-95% accuracy**

2. **Image Quality**: Better quality = better detection
   - High resolution (>512x512): Best
   - Low resolution (<256x256): May be less accurate
   - Compressed/filtered: May affect results

3. **Confidence Threshold**: 
   - Use results >75% confidence for decisions
   - <60% confidence = manual review recommended

### Limitations

- ❌ Model must be trained on real deepfake data for production use
- ❌ New deepfake techniques may not be detected
- ❌ Heavily edited photos may trigger false positives
- ✅ Regular retraining with new data improves accuracy

---

## 🔄 Improving Accuracy

### 1. Train with Real Data
```bash
# Download real deepfake dataset
kaggle datasets download -d xhlulu/140k-real-and-fake-faces

# Organize data
python main.py --prepare --organize_data --source_dir raw_data/

# Train with more data
python main.py --train --model efficientnet --epochs 50 --fine_tune
```

### 2. Use Better Model
```bash
# EfficientNet gives best accuracy
python main.py --train --model efficientnet --epochs 50 --fine_tune
```

### 3. Ensemble Predictions
```python
# Use multiple models and average predictions
models = [
    tf.keras.models.load_model('models/model1.h5'),
    tf.keras.models.load_model('models/model2.h5'),
    tf.keras.models.load_model('models/model3.h5')
]

predictions = [model.predict(img_array)[0][0] for model in models]
final_prediction = np.mean(predictions)
```

---

## 🆘 Troubleshooting

### "Model not found" error
```bash
# Train a model first
python main.py --train --model custom_cnn --epochs 2
```

### "Image not found" error
```bash
# Use absolute path
python check_deepfake.py "C:/Users/You/Pictures/photo.jpg"
```

### Low accuracy on real photos
```bash
# Retrain with real deepfake datasets
# The test model is trained on random data for demonstration only
```

### Memory errors
```bash
# Reduce batch size
python main.py --train --batch_size 16
```

---

## 📞 Quick Reference

| Task | Command |
|------|---------|
| Check single photo | `python check_deepfake.py photo.jpg` |
| Check multiple photos | `python check_deepfake.py *.jpg --batch` |
| Check with custom model | `python check_deepfake.py photo.jpg --model path/to/model.h5` |
| Check directory | `python main.py --inference --directory folder/` |
| Real-time webcam | `python main.py --inference --webcam` |
| Get help | `python check_deepfake.py --help` |

---

**Now you're ready to check any photo for deepfakes! 🎉**

For more advanced usage, see the main [README.md](README.md) and [QUICKSTART.md](QUICKSTART.md).
