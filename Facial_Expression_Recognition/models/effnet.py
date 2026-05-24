from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights
import torch.nn as nn

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=7):

        super().__init__()

        weights = EfficientNet_B0_Weights.DEFAULT
        self.model = efficientnet_b0(weights=weights)

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):
        # EfficientNet expects 3 channels (RGB), triple greyscale?
        x = x.repeat(1, 3, 1, 1)
        return self.model(x)
