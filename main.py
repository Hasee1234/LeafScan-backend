from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "model", "plant_disease_model.keras")
CLASS_PATH = os.path.join(BASE_DIR, "model", "class_names.txt")

# ---------------- LOAD MODEL ----------------
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    model_loaded = True
except Exception as e:
    print("❌ Model load error:", e)
    model = None
    model_loaded = False

# ---------------- LOAD CLASSES ----------------
try:
    with open(CLASS_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except Exception as e:
    print("❌ Class load error:", e)
    class_names = []

# ---------------- HEALTH CHECK ----------------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "classes_count": len(class_names)
    }

# ---------------- GET CLASSES ----------------
@app.get("/classes")
def get_classes():
    return {
        "total": len(class_names),
        "classes": class_names
    }

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))  # adjust if your model uses different size
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# ---------------- PREDICT ----------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model_loaded:
        return {"error": "Model not loaded"}

    try:
        image_bytes = await file.read()
        img = preprocess_image(image_bytes)

        predictions = model.predict(img)
        predicted_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        predicted_class = class_names[predicted_index] if class_names else "Unknown"

        top_5_idx = np.argsort(predictions[0])[-5:][::-1]
        top_5 = {
            class_names[i] if i < len(class_names) else str(i): float(predictions[0][i])
            for i in top_5_idx
        }

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_5": top_5
        }

    except Exception as e:
        return {"error": str(e)}