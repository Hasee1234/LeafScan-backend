# """
# LeafScan — EfficientNetB0 Training Script
# ==========================================
# Dataset  : PlantVillage (auto-downloaded via TensorFlow Datasets)
# Model    : EfficientNetB0 with Transfer Learning + Fine-tuning
# Python   : 3.11
# TF       : 2.19.0
# Output   : model/plant_disease_model.keras
# """

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import tensorflow_datasets as tfds

# tf.keras.mixed_precision.set_global_policy("float32")
# # ── Silence TF logs (optional) ──
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# print(f"✅ TensorFlow version : {tf.__version__}")
# print(f"✅ NumPy version      : {np.__version__}")

# # ══════════════════════════════════════════
# # 1. CONFIG
# # ══════════════════════════════════════════
# IMG_SIZE    = 224
# BATCH_SIZE  = 16
# NUM_CLASSES = 38
# EPOCHS_1    = 10   # Phase 1 — frozen base (feature extraction)
# EPOCHS_2    = 10   # Phase 2 — unfrozen top layers (fine-tuning)
# MODEL_DIR   = "model"
# MODEL_PATH  = os.path.join(MODEL_DIR, "plant_disease_model.keras")

# os.makedirs(MODEL_DIR, exist_ok=True)

# # ══════════════════════════════════════════
# # 2. LOAD DATASET via TensorFlow Datasets
# #    Auto-downloads PlantVillage on first run
# #    (~800 MB) — no Kaggle account needed
# # ══════════════════════════════════════════
# print("\n📥 Loading PlantVillage dataset...")

# (ds_train, ds_val), ds_info = tfds.load(
#     "plant_village",
#     split=["train[:80%]", "train[80%:]"],
#     as_supervised=True,
#     with_info=True,
#     shuffle_files=True,
# )

# print(f"✅ Dataset loaded")
# print(f"   Classes  : {ds_info.features['label'].num_classes}")
# print(f"   Train    : {ds_train.cardinality()} batches approx")

# # Class names
# CLASS_NAMES = ds_info.features["label"].names
# print(f"\n📋 Class names ({len(CLASS_NAMES)} total):")
# for i, name in enumerate(CLASS_NAMES):
#     print(f"   {i:2d}. {name}")

# # Save class names to file for use in main.py
# with open(os.path.join(MODEL_DIR, "class_names.txt"), "w") as f:
#     for name in CLASS_NAMES:
#         f.write(name + "\n")
# print(f"\n✅ Class names saved to {MODEL_DIR}/class_names.txt")

# # ══════════════════════════════════════════
# # 3. PREPROCESSING & AUGMENTATION
# # ══════════════════════════════════════════

# # EfficientNetB0 expects pixel values in [0, 255]
# # so we do NOT normalize — just resize and cast
# def preprocess(image, label):
#     image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
#     image = tf.cast(image, tf.float32)   # keep in [0, 255]
#     label = tf.one_hot(label, NUM_CLASSES)
#     return image, label

# def augment(image, label):
#     image, label = preprocess(image, label)
#     image = tf.image.random_flip_left_right(image)
#     image = tf.image.random_flip_up_down(image)
#     image = tf.image.random_brightness(image, max_delta=0.2)
#     image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
#     image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
#     image = tf.clip_by_value(image, 0, 255)
#     return image, label

# AUTOTUNE = tf.data.AUTOTUNE

# ds_train = (
#     ds_train
#     .map(augment,    num_parallel_calls=AUTOTUNE)
#     .cache()
#     .shuffle(1000)
#     .batch(BATCH_SIZE)
#     .prefetch(AUTOTUNE)
# )

# ds_val = (
#     ds_val
#     .map(preprocess, num_parallel_calls=AUTOTUNE)
#     .cache()
#     .batch(BATCH_SIZE)
#     .prefetch(AUTOTUNE)
# )

# print("\n✅ Data pipeline ready")

# # ══════════════════════════════════════════
# # 4. BUILD MODEL
# #    EfficientNetB0 — Transfer Learning
# # ══════════════════════════════════════════
# print("\n🧠 Building EfficientNetB0 model...")

# # Load EfficientNetB0 — pretrained on ImageNet
# # include_top=False removes the 1000-class classifier
# base_model = tf.keras.applications.EfficientNetB0(
#     input_shape=(IMG_SIZE, IMG_SIZE, 3),
#     include_top=False,
#     weights="imagenet",
# )

# # Phase 1: Freeze entire base — only train custom head
# base_model.trainable = False

# # Build custom classification head
# inputs  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
# x       = base_model(inputs, training=False)
# x       = tf.keras.layers.GlobalAveragePooling2D()(x)
# x       = tf.keras.layers.BatchNormalization()(x)
# x       = tf.keras.layers.Dropout(0.3)(x)
# x       = tf.keras.layers.Dense(256, activation="relu")(x)
# x       = tf.keras.layers.BatchNormalization()(x)
# x       = tf.keras.layers.Dropout(0.2)(x)
# outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

# model = tf.keras.Model(inputs, outputs, name="LeafScan_EfficientNetB0")

# model.summary()

# # ══════════════════════════════════════════
# # 5. PHASE 1 — FEATURE EXTRACTION
# #    Train only the custom head
# # ══════════════════════════════════════════
# print("\n🚀 Phase 1: Feature Extraction (frozen base)...")

# tf.keras.backend.set_floatx('float32')
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"],
# )

# callbacks_phase1 = [
#     tf.keras.callbacks.EarlyStopping(
#         monitor="val_accuracy",
#         patience=3,
#         restore_best_weights=True,
#         verbose=1,
#     ),
#     tf.keras.callbacks.ReduceLROnPlateau(
#         monitor="val_loss",
#         factor=0.5,
#         patience=2,
#         min_lr=1e-7,
#         verbose=1,
#     ),
#     tf.keras.callbacks.ModelCheckpoint(
#         filepath=os.path.join(MODEL_DIR, "checkpoint_phase1.keras"),
#         save_best_only=True,
#         monitor="val_accuracy",
#         verbose=1,
#     ),
# ]

# history1 = model.fit(
#     ds_train,
#     validation_data=ds_val,
#     epochs=EPOCHS_1,
#     callbacks=callbacks_phase1,
# )

# print(f"\n✅ Phase 1 complete")
# print(f"   Best val accuracy: {max(history1.history['val_accuracy']):.4f}")

# # ══════════════════════════════════════════
# # 6. PHASE 2 — FINE-TUNING
# #    Unfreeze top layers of base model
# # ══════════════════════════════════════════
# print("\n🔧 Phase 2: Fine-tuning (unfreezing top 30 layers)...")

# base_model.trainable = True

# # Freeze all layers EXCEPT the last 30
# for layer in base_model.layers[:-30]:
#     layer.trainable = False

# print(f"   Trainable layers: {sum(1 for l in model.layers if l.trainable)}")

# # Recompile with lower learning rate for fine-tuning
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"],
# )

# callbacks_phase2 = [
#     tf.keras.callbacks.EarlyStopping(
#         monitor="val_accuracy",
#         patience=5,
#         restore_best_weights=True,
#         verbose=1,
#     ),
#     tf.keras.callbacks.ReduceLROnPlateau(
#         monitor="val_loss",
#         factor=0.3,
#         patience=2,
#         min_lr=1e-8,
#         verbose=1,
#     ),
#     tf.keras.callbacks.ModelCheckpoint(
#         filepath=os.path.join(MODEL_DIR, "checkpoint_phase2.keras"),
#         save_best_only=True,
#         monitor="val_accuracy",
#         verbose=1,
#     ),
# ]

# history2 = model.fit(
#     ds_train,
#     validation_data=ds_val,
#     epochs=EPOCHS_2,
#     callbacks=callbacks_phase2,
# )

# print(f"\n✅ Phase 2 complete")
# print(f"   Best val accuracy: {max(history2.history['val_accuracy']):.4f}")

# # ══════════════════════════════════════════
# # 7. SAVE FINAL MODEL
# # ══════════════════════════════════════════
# model.save(MODEL_PATH)
# print(f"\n✅ Model saved to: {MODEL_PATH}")

# # ══════════════════════════════════════════
# # 8. PLOT TRAINING CURVES
# # ══════════════════════════════════════════
# def plot_history(h1, h2):
#     acc     = h1.history["accuracy"]     + h2.history["accuracy"]
#     val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
#     loss    = h1.history["loss"]         + h2.history["loss"]
#     val_loss= h1.history["val_loss"]     + h2.history["val_loss"]
#     epochs  = range(1, len(acc) + 1)
#     phase2_start = len(h1.history["accuracy"]) + 1

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

#     ax1.plot(epochs, acc,     label="Train Accuracy",      color="#52b788")
#     ax1.plot(epochs, val_acc, label="Val Accuracy",        color="#1a3d2b")
#     ax1.axvline(x=phase2_start, color="orange", linestyle="--", label="Fine-tuning starts")
#     ax1.set_title("LeafScan — Accuracy", fontsize=13)
#     ax1.set_xlabel("Epoch")
#     ax1.set_ylabel("Accuracy")
#     ax1.legend()
#     ax1.grid(alpha=0.3)

#     ax2.plot(epochs, loss,     label="Train Loss",         color="#52b788")
#     ax2.plot(epochs, val_loss, label="Val Loss",           color="#1a3d2b")
#     ax2.axvline(x=phase2_start, color="orange", linestyle="--", label="Fine-tuning starts")
#     ax2.set_title("LeafScan — Loss", fontsize=13)
#     ax2.set_xlabel("Epoch")
#     ax2.set_ylabel("Loss")
#     ax2.legend()
#     ax2.grid(alpha=0.3)

#     plt.tight_layout()
#     plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"), dpi=150)
#     print(f"✅ Training curves saved to {MODEL_DIR}/training_curves.png")
#     plt.show()

# plot_history(history1, history2)

# # ══════════════════════════════════════════
# # 9. FINAL EVALUATION
# # ══════════════════════════════════════════
# print("\n📊 Final Evaluation on Validation Set:")
# results = model.evaluate(ds_val, verbose=1)
# print(f"\n   Loss     : {results[0]:.4f}")
# print(f"   Accuracy : {results[1]:.4f} ({results[1]*100:.2f}%)")
# print(f"   Top-5 Acc: {results[2]:.4f} ({results[2]*100:.2f}%)")
# print("\n🌿 Training complete! Model is ready for deployment.")



"""
LeafScan — EfficientNetB0 Training Script (FIXED)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

# ✅ MUST BE FIRST (very important fix)
tf.keras.mixed_precision.set_global_policy("float32")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print(f"✅ TensorFlow version : {tf.__version__}")
print(f"✅ NumPy version      : {np.__version__}")

# ══════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════
IMG_SIZE    = 224
BATCH_SIZE  = 16
NUM_CLASSES = 38
EPOCHS_1    = 10
EPOCHS_2    = 10
MODEL_DIR   = "model"
MODEL_PATH  = os.path.join(MODEL_DIR, "plant_disease_model.keras")

os.makedirs(MODEL_DIR, exist_ok=True)

# ══════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════
print("\n📥 Loading dataset...")

(ds_train, ds_val), ds_info = tfds.load(
    "plant_village",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    with_info=True,
    shuffle_files=True,
)

CLASS_NAMES = ds_info.features["label"].names

def preprocess(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)

    # ✅ force float32 output explicitly (important fix)
    label = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
    return image, label

def augment(image, label):
    image, label = preprocess(image, label)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.clip_by_value(image, 0, 255)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE

ds_train = ds_train.map(augment, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
ds_val   = ds_val.map(preprocess, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

print("✅ Data ready")

# ══════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.summary()

# ══════════════════════════════════════════
# PHASE 1
# ══════════════════════════════════════════
print("\n🚀 Phase 1")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, dtype=tf.float32)
    ],
)

history1 = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS_1)

print("✅ Phase 1 done")

# ══════════════════════════════════════════
# PHASE 2
# ══════════════════════════════════════════
print("\n🔧 Phase 2")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, dtype=tf.float32)
    ],
)

history2 = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS_2)

# ══════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════
model.save(MODEL_PATH)
print("✅ Saved model")

print("\n🌿 Training complete")