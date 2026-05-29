from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os
from tensorflow.keras.applications.efficientnet import preprocess_input

app = FastAPI(
    title="LeafScan API",
    description="AI-powered plant disease detection using EfficientNetB0",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://leaf-scan-eight.vercel.app/detect"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
CLASS_NAMES = []

# Paths
MODEL_PATH = os.getenv("MODEL_PATH", "model/plant_disease_model.keras")
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", "model/class_names.txt")


# Load model and classes on startup
@app.on_event("startup")
async def load_model():
    global MODEL, CLASS_NAMES

    print("\n========== STARTUP ==========")
    print("Loading model from:", MODEL_PATH)
    print("Loading classes from:", CLASS_NAMES_PATH)

    # Load class names
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r") as f:
            CLASS_NAMES = [line.strip() for line in f if line.strip()]
        print(f"✅ Loaded {len(CLASS_NAMES)} class names")
    else:
        print("❌ class_names.txt not found")

    # Load model
    if os.path.exists(MODEL_PATH):
        print(f"📦 Loading model...")
        MODEL = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully")
    else:
        print("⚠️ Model not found")

    # 🔥 DEBUG OUTPUT (IMPORTANT)
    print("\n========== DEBUG CLASS INFO ==========")
    print("TOTAL CLASSES:", len(CLASS_NAMES))
    print("FIRST 10 CLASSES:", CLASS_NAMES[:10])
    print("LAST 10 CLASSES:", CLASS_NAMES[-10:])
    print("======================================\n")


# Image preprocessing
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# Root
@app.get("/")
def root():
    return {
        "message": "LeafScan API is running 🌿",
        "model_loaded": MODEL is not None,
        "classes_loaded": len(CLASS_NAMES),
    }


# Health
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "classes_count": len(CLASS_NAMES),
    }


# Classes
@app.get("/classes")
def get_classes():
    return {
        "total": len(CLASS_NAMES),
        "classes": CLASS_NAMES
    }


# Predict
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if MODEL is None:
        return {"error": "Model not loaded"}

    # Read image
    contents = await file.read()

    # Preprocess
    img_array = preprocess_image(contents)

    # Predict
    predictions = MODEL.predict(img_array, verbose=0)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    class_name = CLASS_NAMES[predicted_index]

    # 🔥 DEBUG PREDICTION
    print("\n========== PREDICTION DEBUG ==========")
    print("Predicted index:", predicted_index)
    print("Predicted class:", class_name)
    print("Top 5 indices:", np.argsort(predictions[0])[-5:])
    print("Top 5 values:", sorted(predictions[0])[-5:])
    print("======================================\n")

    return {
        "class": class_name,
        "confidence": confidence
    }