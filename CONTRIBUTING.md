# Contributing to Deepfake Detection System

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## 🤝 How to Contribute

### Types of Contributions

We welcome:
- 🐛 Bug reports and fixes
- ✨ New features and enhancements
- 📝 Documentation improvements
- 🧪 Test coverage improvements
- 🎨 Code quality improvements
- 💡 Ideas and suggestions

## 🚀 Getting Started

### 1. Fork the Repository

Click the "Fork" button on GitHub to create your own copy.

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/DeepFake-Detection-for-Image.git
cd DeepFake-Detection-for-Image
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

Branch naming convention:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation
- `test/` - Tests
- `refactor/` - Code improvements

### 4. Set Up Development Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📝 Development Guidelines

### Code Style

- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions small and focused

Example:
```python
def detect_deepfake(image_path):
    """
    Detect if an image is a deepfake.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Prediction results with probability and class
        
    Raises:
        ValueError: If image_path is invalid
    """
    # Implementation here
    pass
```

### Documentation

- Update README.md if adding new features
- Add comments for complex logic
- Update QUICKSTART.md for user-facing changes
- Include docstrings in all modules

### Testing

Before submitting:
```bash
# Test your changes
python main.py --test

# Test preprocessing
cd src
python preprocess.py

# Test model creation
python model.py

# Test training (with sample data)
python train.py --epochs 2
```

## 🔧 Making Changes

### Adding a New Model Architecture

1. Edit `src/model.py`
2. Add your model creation function:

```python
@staticmethod
def create_your_model(input_shape):
    """Create your custom model"""
    model = models.Sequential(name='Your_Model')
    # Add layers
    return model
```

3. Update the `get_model()` function to include your model
4. Test the model creation
5. Update documentation

### Adding Preprocessing Features

1. Edit `src/preprocess.py`
2. Add your preprocessing function:

```python
def your_preprocessing_function(self, image):
    """
    Your preprocessing description
    
    Args:
        image: Input image
        
    Returns:
        Processed image
    """
    # Implementation
    return processed_image
```

3. Update relevant methods
4. Add tests

### Improving Training

1. Edit `src/train.py`
2. Add callbacks, optimizers, or training strategies
3. Test with small dataset
4. Document new parameters

## 📤 Submitting Changes

### 1. Commit Your Changes

```bash
git add .
git commit -m "Add: Brief description of changes"
```

Commit message format:
- `Add:` - New features
- `Fix:` - Bug fixes
- `Update:` - Updates to existing features
- `Docs:` - Documentation changes
- `Test:` - Test additions/changes
- `Refactor:` - Code improvements

### 2. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 3. Create Pull Request

1. Go to the original repository
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill in the PR template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring

## Testing
- [ ] Tested locally
- [ ] All tests pass
- [ ] Documentation updated

## Screenshots (if applicable)
Add screenshots here
```

## 🐛 Reporting Bugs

### Before Reporting

1. Check existing issues
2. Verify it's a bug (not a question)
3. Test with the latest version

### Bug Report Template

```markdown
**Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Run '...'
3. See error

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python: [e.g., 3.9.5]
- TensorFlow: [e.g., 2.13.0]

**Error Messages**
```
Paste error messages here
```

**Screenshots**
If applicable
```

## 💡 Suggesting Features

### Feature Request Template

```markdown
**Is your feature related to a problem?**
Description of the problem

**Describe the solution**
How you'd like it to work

**Describe alternatives**
Alternative solutions considered

**Additional context**
Any other context or screenshots
```

## 🧪 Testing Guidelines

### Unit Tests

Add tests for new functions:

```python
def test_your_function():
    """Test your function"""
    result = your_function(test_input)
    assert result == expected_output
```

### Integration Tests

Test complete workflows:

```python
def test_training_pipeline():
    """Test complete training pipeline"""
    trainer = ModelTrainer('custom_cnn')
    # Test with minimal data
    history = trainer.train_with_arrays(X_train, y_train, X_val, y_val, epochs=1)
    assert history is not None
```

## 📋 Code Review Process

### What We Look For

1. **Functionality**: Does it work as intended?
2. **Code Quality**: Is it clean and maintainable?
3. **Documentation**: Is it well-documented?
4. **Testing**: Are there adequate tests?
5. **Performance**: Does it impact performance negatively?

### Review Timeline

- Initial review: 1-3 days
- Follow-up reviews: 1-2 days
- Merge: After approval from maintainer

## 🎯 Priority Areas

We especially welcome contributions in:

1. **New Model Architectures**
   - Xception, ResNet, Vision Transformers
   - Ensemble methods

2. **Preprocessing Improvements**
   - Better face detection
   - Advanced augmentation techniques
   - Video frame extraction

3. **Deployment Features**
   - Web API (Flask/FastAPI)
   - Mobile app integration
   - Cloud deployment guides

4. **Explainability**
   - Grad-CAM visualization
   - LIME/SHAP integration
   - Attention maps

5. **Performance Optimization**
   - Model quantization
   - TensorRT integration
   - Faster inference

## 📚 Resources

### Learning Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Guide](https://keras.io/guides/)
- [Python Style Guide (PEP 8)](https://pep8.org/)
- [Git Guide](https://git-scm.com/doc)

### Related Papers

- EfficientNet: Rethinking Model Scaling for CNNs
- MesoNet: Compact Facial Video Forgery Detection
- FaceForensics++: Learning to Detect Manipulated Faces

## 💬 Communication

### Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, showcases
- **Pull Requests**: Code contributions

### Best Practices

- Be respectful and constructive
- Provide context and examples
- Be patient with reviews
- Ask questions if unclear

## ✅ Checklist Before Submitting

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] Tests pass locally
- [ ] Commit messages are clear
- [ ] PR description is complete

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

## 📄 License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

## 🎉 Thank You!

Every contribution, no matter how small, makes this project better. Thank you for taking the time to contribute!

---

**Questions?** Open an issue or discussion on GitHub.

**Ready to contribute?** Fork, code, and submit a PR!
