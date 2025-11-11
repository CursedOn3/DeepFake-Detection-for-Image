"""
Deepfake Detection System
A comprehensive deep learning system for detecting deepfake images
"""

__version__ = "1.0.0"
__author__ = "CursedOn3"
__email__ = "your.email@example.com"

from . import config
from . import preprocess
from . import model
from . import train
from . import evaluate
from . import inference

__all__ = [
    'config',
    'preprocess',
    'model',
    'train',
    'evaluate',
    'inference'
]
