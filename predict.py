"""Run inference with a trained Keras image classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf


def predict_image(
    model: tf.keras.Model, image_path: Path, image_size: tuple[int, int]
) -> np.ndarray:
    """Return class probabilities for one image without modifying the file."""
    with Image.open(image_path) as image:
        image_array = np.asarray(image.convert("RGB").resize(image_size), dtype=np.float32)
    return model.predict(image_array[None, ...] / 255.0, verbose=0)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--class-indices", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"Image does not exist: {args.image}")
    model = tf.keras.models.load_model(args.model, compile=False)
    class_indices = json.loads(args.class_indices.read_text(encoding="utf-8"))
    index_to_class = {index: name for name, index in class_indices.items()}
    probabilities = predict_image(model, args.image, (args.image_size, args.image_size))
    predicted_index = int(np.argmax(probabilities))
    print(
        json.dumps(
            {
                "class": index_to_class[predicted_index],
                "confidence": float(probabilities[predicted_index]),
                "probabilities": {
                    index_to_class[index]: float(probability)
                    for index, probability in enumerate(probabilities)
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
