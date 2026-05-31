import os

import torch

from Facial_Expression_Recognition.models.cnn import CNNImageClassifier
from Facial_Expression_Recognition.models.effnet import EfficientNetClassifier


def load_cnn(device):
    model = CNNImageClassifier(num_classes=7).to(device)
    state = torch.load(os.getenv("CNN_WEIGHTS", "models/baseline_cnn_best_weights.pth"), map_location=device)
    try:
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model weights: {e}")
    return model


def load_effnet(device):
    model = EfficientNetClassifier(num_classes=7).to(device)
    state = torch.load(
        os.getenv("EFFNET_WEIGHTS", "models/effnet.pth"), map_location=device
    )
    try:
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model weights: {e}")

    return model
