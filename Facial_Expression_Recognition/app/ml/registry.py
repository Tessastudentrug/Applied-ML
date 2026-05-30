from Facial_Expression_Recognition.app.ml.loader import load_cnn, load_effnet


class ModelRegistry:
    def __init__(self):
        self.models = {}
        self.device = None

    def load(self, device):
        self.device = device

        self.models["cnn"] = load_cnn(device)
        self.models["effnet"] = load_effnet(device)

    def get(self, model_id: str):
        return self.models[model_id]

    def list(self):
        return list(self.models.keys())

    def clear(self):
        self.models.clear()
