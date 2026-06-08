"""
EfficientNet-B0 based facial expression classifier.

Uses a pretrained EfficientNet backbone with a custom classification head
adapted for emotion recognition.
"""

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B0 model adapted for grayscale facial emotion classification.

    The model:
    - Loads pretrained ImageNet weights
    - Replaces classifier head for num_classes emotions
    - Handles grayscale input by converting to 3-channel format
    """

    def __init__(self, num_classes=7, dropout=0.5):
        super().__init__()

        weights = EfficientNet_B0_Weights.DEFAULT
        self.model = efficientnet_b0(weights=weights)

        # Setup the custom classifier with the Dropout restored
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(p=dropout), 
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, H, W) in grayscale.

        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        # EfficientNet expects 3 channels (RGB),
        # so we copy the grayscale channel 3 times
        x = x.repeat(1, 3, 1, 1)
        return self.model(x)
