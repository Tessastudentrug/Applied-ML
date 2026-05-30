import torch

from Facial_Expression_Recognition.app.config import (
    CNN_WEIGHTS,
    EFFNET_WEIGHTS,
)
from Facial_Expression_Recognition.models.cnn import CNNImageClassifier
from Facial_Expression_Recognition.models.effnet import EfficientNetClassifier


def load_cnn(device):
    model = CNNImageClassifier(num_classes=7).to(device)
    state = torch.load(CNN_WEIGHTS, map_location=device)
    try:
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model weights: {e}")
    return model


def load_effnet(device):
    model = EfficientNetClassifier(num_classes=7).to(device)
    state = torch.load(EFFNET_WEIGHTS, map_location=device)
    try:
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model weights: {e}")

    return model
