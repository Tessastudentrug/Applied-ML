from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware

# Import your model architectures and the EVALUATION transform
from models.effnet import EfficientNetClassifier
from models.cnn import CNNImageClassifier
from features.preprocessing import get_eval_transform

# Initialize the API with documentation metadata
app = FastAPI(
    title="Facial Expression Recognition API", 
    description="RESTful API for classifying emotions from face images."
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows your local HTML file to talk to the API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 1. Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Initialize and load the model (Stateless & Cacheable)
model = EfficientNetClassifier(num_classes=7).to(device)

# NOTE: Update this path to point to your actual trained weights!
MODEL_PATH = "../models/effnet.pth" 

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # CRITICAL: Put model in evaluation mode
except Exception as e:
    print(f"Failed to load model weights: {e}")

# 3. Setup preprocessing and labels
# We strictly use get_eval_transform so the API doesn't randomly flip images!
transform = get_eval_transform(image_size=64)
emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    """
    Accepts an image upload and returns the predicted emotion.
    """
    # Use HTTP 400 (Bad Request) if the user uploads a non-image file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # Read the file bytes
        contents = await file.read()
        
        # Convert to Grayscale ("L") to match your training pipeline
        image = Image.open(io.BytesIO(contents)).convert("L")
        
        # Apply the eval transform and add batch dimension
        tensor = transform(image).unsqueeze(0).to(device)
        
        # Make the prediction without tracking gradients
        with torch.no_grad():
            outputs = model(tensor)
            _, predicted_idx = torch.max(outputs, 1)
            
        emotion = emotion_labels[predicted_idx.item()]
        
        # Return a clean JSON response (HTTP 200 is default for successful FastAPI returns)
        return {
            "filename": file.filename,
            "predicted_emotion": emotion,
            "status_code": 200
        }
        
    except Exception as e:
        # Use HTTP 500 (Internal Server Error) if the processing crashes
        raise HTTPException(status_code=500, detail=f"An error occurred processing the image: {str(e)}")