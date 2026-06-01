"""
Model registry for managing loaded ML models.

Handles loading, storage, and retrieval of different facial expression
recognition models (e.g., CNN and EfficientNet).
"""

from Facial_Expression_Recognition.app.ml.loader import load_cnn, load_effnet


class ModelRegistry:
    """
    Central registry for ML models used in inference.

    Responsible for:
    - Loading models onto the correct device (CPU/GPU)
    - Storing loaded models in memory
    - Providing access to models by ID
    """

    def __init__(self):
        self.models = {}
        self.device = None

    def load(self, device):
        """
        Load all supported models into memory.

        Args:
            device: Torch device (CPU or CUDA) used for inference.
        """
        self.device = device

        self.models["cnn"] = load_cnn(device)
        self.models["effnet"] = load_effnet(device)

    def get(self, model_id: str):
        """
        Retrieve a loaded model by its ID.

        Args:
            model_id: Name of the model (e.g., 'cnn', 'effnet').

        Returns:
            Loaded PyTorch model.
        """
        return self.models[model_id]

    def list(self):
        """
        List all loaded model IDs.

        Returns:
            List of available model names.
        """
        return list(self.models.keys())

    def clear(self):
        """
        Clear all loaded models from memory.
        """
        self.models.clear()
