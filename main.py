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
    allow_origins=["*"],  # change later after deployment if needed
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

# Disease mapping
CLASS_TO_DISEASE_ID = {
    "Apple___Apple_scab": "apple-scab",
    "Apple___Black_rot": "apple-black-rot",
    "Apple___Cedar_apple_rust": "apple-cedar-rust",
    "Apple___healthy": "healthy",

    "Blueberry___healthy": "healthy",

    "Cherry___healthy": "healthy",
    "Cherry___Powdery_mildew": "cherry-powdery-mildew",

    "Corn___Cercospora_leaf_spot Gray_leaf_spot": "corn-gray-leaf-spot",
    "Corn___Common_rust": "corn-common-rust",
    "Corn___Northern_Leaf_Blight": "corn-northern-leaf-blight",
    "Corn___healthy": "healthy",

    "Grape___Black_rot": "grape-black-rot",
    "Grape___Esca_(Black_Measles)": "grape-esca",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "grape-leaf-blight",
    "Grape___healthy": "healthy",

    "Orange___Haunglongbing_(Citrus_greening)": "orange-citrus-greening",

    "Peach___Bacterial_spot": "peach-bacterial-spot",
    "Peach___healthy": "healthy",

    "Pepper,_bell___Bacterial_spot": "pepper-bacterial-spot",
    "Pepper,_bell___healthy": "healthy",

    "Potato___Early_blight": "potato-early-blight",
    "Potato___Late_blight": "potato-late-blight",
    "Potato___healthy": "healthy",

    "Raspberry___healthy": "healthy",

    "Soybean___healthy": "healthy",

    "Squash___Powdery_mildew": "squash-powdery-mildew",

    "Strawberry___Leaf_scorch": "strawberry-leaf-scorch",
    "Strawberry___healthy": "healthy",

    "Tomato___Bacterial_spot": "tomato-bacterial-spot",
    "Tomato___Early_blight": "tomato-early-blight",
    "Tomato___Late_blight": "tomato-late-blight",
    "Tomato___Leaf_Mold": "tomato-leaf-mold",
    "Tomato___Septoria_leaf_spot": "tomato-septoria-leaf-spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "tomato-spider-mites",
    "Tomato___Target_Spot": "tomato-target-spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato-yellow-leaf-curl-virus",
    "Tomato___Tomato_mosaic_virus": "tomato-mosaic-virus",
    "Tomato___healthy": "healthy",
}


# Load model and classes on startup
@app.on_event("startup")
async def load_model():
    global MODEL, CLASS_NAMES

    # Load class names
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r") as f:
            CLASS_NAMES = [line.strip() for line in f if line.strip()]

        print(f"✅ Loaded {len(CLASS_NAMES)} class names")

    else:
        print("❌ class_names.txt not found")

    # Load model
    if os.path.exists(MODEL_PATH):
        print(f"📦 Loading model from {MODEL_PATH}...")

        MODEL = tf.keras.models.load_model(MODEL_PATH)

        print("✅ Model loaded successfully")

    else:
        print("⚠️ Model not found — running in demo mode")


# Image preprocessing
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img = img.resize((224, 224))

    img_array = np.array(img, dtype=np.float32)

    # EfficientNet preprocessing
    img_array = preprocess_input(img_array)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# Root route
@app.get("/")
def root():
    return {
        "message": "LeafScan API is running 🌿",
        "model_loaded": MODEL is not None,
        "classes_loaded": len(CLASS_NAMES),
        "version": "1.0.0"
    }


# Health check
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "classes_count": len(CLASS_NAMES),
    }


# Get all classes
@app.get("/classes")
def get_classes():
    return {
        "total": len(CLASS_NAMES),
        "classes": CLASS_NAMES
    }


# Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Validate image
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    # Read image
    contents = await file.read()

    # File size check
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Max 10MB."
        )

    # Demo mode if model missing
    if MODEL is None:
        return {
            "disease_id": "tomato-early-blight",
            "disease_name": "Tomato Early Blight",
            "class_name": "Tomato___Early_blight",
            "confidence": 0.923,
            "plant": "Tomato",
            "demo_mode": True,
        }

    try:
        # Preprocess image
        img_array = preprocess_image(contents)

        # Predict
        predictions = MODEL.predict(img_array, verbose=0)

        predicted_index = int(np.argmax(predictions[0]))

        confidence = float(np.max(predictions[0]))

        class_name = CLASS_NAMES[predicted_index]

        # Confidence threshold
        if confidence < 0.65:
            return {
                "disease_id": "unknown",
                "disease_name": "Uncertain Prediction",
                "class_name": "unknown",
                "confidence": round(confidence, 4),
                "plant": "unknown",
                "message": "Please upload a clearer image of a supported plant leaf.",
                "demo_mode": False,
            }

        # Debug logs
        print("\n================ PREDICTION ================")
        print("Predicted index:", predicted_index)
        print("Class name:", class_name)
        print("Confidence:", confidence)

        top5_idx = np.argsort(predictions[0])[-5:][::-1]

        print("\nTop 5 Predictions:")

        for i in top5_idx:
            print(
                f"{CLASS_NAMES[i]} -> {float(predictions[0][i]):.4f}"
            )

        print("===========================================\n")

        # Map disease ID
        disease_id = CLASS_TO_DISEASE_ID.get(class_name, "unknown")

        # Extract plant name
        plant = class_name.split("___")[0].replace("_", " ").strip()

        # Disease readable name
        disease_name = (
            class_name.split("___")[1]
            .replace("_", " ")
            .strip()
        )

        return {
            "disease_id": disease_id,
            "disease_name": disease_name,
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "plant": plant,
            "demo_mode": False,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


