import tensorflow as tf
import numpy as np
from PIL import Image

CLASS_NAMES = ["PCOS", "Normal"]

def build_model(num_classes):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3)
    )

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)

# LOAD MODEL (WEIGHTS ONLY)
model = build_model(len(CLASS_NAMES))
model.load_weights("models/weights.weights.h5")

print("Model loaded successfully")

# IMAGE
img = Image.open("test_image.jpg").convert("RGB")
img = img.resize((224, 224))

img = np.array(img) / 255.0
img = np.expand_dims(img, axis=0)

# PREDICT
pred = model.predict(img)

print("Prediction:", CLASS_NAMES[np.argmax(pred)])
print("Confidence:", np.max(pred))