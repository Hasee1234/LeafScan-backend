import os
import traceback
from fastapi import FastAPI, File, UploadFile
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = FastAPI()

# =============================
# PATH SETUP (IMPORTANT)
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CHANGE THIS depending on your repo file
MODEL_PATH = os.path.join(BASE_DIR, "model", "plant_disease_model.keras")
CLASS_PATH = os.path.join(BASE_DIR, "model", "class_names.txt")

print("BASE_DIR:", BASE_DIR)
print("MODEL_PATH:", MODEL_PATH)
print("CLASS_PATH:", CLASS_PATH)

# =============================
# GLOBALS
# =============================
model = None
classes = []

# =============================
# LOAD MODEL ON STARTUP
# =============================
try:
    print("Loading model...")

    model = load_model(MODEL_PATH)
    print("Model loaded successfully ✔")

    with open(CLASS_PATH, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    print("Classes loaded:", len(classes))

except Exception as e:
    print("\n❌ MODEL LOAD FAILED ❌")
    traceback.print_exc()


# =============================
# HEALTH CHECK
# =============================
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes_count": len(classes)
    }


# =============================
# GET CLASSES
# =============================
@app.get("/classes")
def get_classes():
    return {
        "total": len(classes),
        "classes": classes
    }


# =============================
# PREDICT ENDPOINT
# =============================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if model is None:
        return {"error": "Model not loaded on server"}

    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Resize (IMPORTANT: must match training size)
        image = image.resize((224, 224))

        # Convert to array
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)

        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return {
            "class": classes[class_index] if class_index < len(classes) else "unknown",
            "confidence": confidence
        }

    except Exception as e:
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }