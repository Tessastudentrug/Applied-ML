"""
Model loading utilities for CNN and EfficientNet classifiers.

This module loads pretrained PyTorch models, moves them to the correct device,
and prepares them for inference mode.
"""

import os

import torch

from Facial_Expression_Recognition.app.config import CNN_WEIGHTS, EFFNET_WEIGHTS
from Facial_Expression_Recognition.models.cnn import CNNImageClassifier
from Facial_Expression_Recognition.models.effnet import EfficientNetClassifier


def load_cnn(device):
    """
    Load the CNN-based facial expression classifier.

    Args:
        device: Torch device (CPU or CUDA).

    Returns:
        CNNImageClassifier model in evaluation mode.
    """
    model = CNNImageClassifier(num_classes=7).to(device)
    state = torch.load(os.getenv("CNN_WEIGHTS", CNN_WEIGHTS), map_location=device)
    try:
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load CNN model weights: {e}")
    return model


def load_effnet(device):
    """
    Load the EfficientNet-based facial expression classifier.

    Args:
        device: Torch device (CPU or CUDA).

    Returns:
        EfficientNetClassifier model in evaluation mode.
    """
    model = EfficientNetClassifier(num_classes=7).to(device)
    state = torch.load(os.getenv("EFFNET_WEIGHTS", EFFNET_WEIGHTS), map_location=device)
    try:
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load EfficientNet model weights: {e}")

    return model
