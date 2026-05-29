import os
from fastapi import FastAPI, File, UploadFile
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = FastAPI()

# BASE PATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "plant_disease_model.keras")
CLASS_PATH = os.path.join(BASE_DIR, "model", "class_names.txt")

# LOAD MODEL
model = None
classes = []

try:
    model = load_model(MODEL_PATH)

    with open(CLASS_PATH, "r") as f:
        classes = [line.strip() for line in f.readlines()]

except Exception as e:
    print("ERROR LOADING MODEL:", e)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes_count": len(classes)
    }


@app.get("/classes")
def get_classes():
    return {
        "total": len(classes),
        "classes": classes
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}

    image = Image.open(io.BytesIO(await file.read()))
    image = image.resize((224, 224))  # adjust if needed

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    return {
        "class": classes[class_index],
        "confidence": float(np.max(prediction))
    }