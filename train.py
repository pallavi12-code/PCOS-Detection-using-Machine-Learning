"""Train the PCOS ultrasound classifier.

The script is intentionally environment-agnostic: paths are supplied through
CLI arguments instead of relying on Google Colab or mounted Drive folders.

Example:
    python train.py --data-dir data/ultrasound --output-dir artifacts
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetB0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def set_seed(seed: int) -> None:
    """Make Python, NumPy and TensorFlow operations as reproducible as practical."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def discover_samples(data_dir: Path) -> tuple[list[tuple[str, int]], dict[str, int]]:
    """Discover image files from one subdirectory per class."""
    class_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if len(class_dirs) < 2:
        raise ValueError("data_dir must contain at least two class directories")

    class_indices = {directory.name: index for index, directory in enumerate(class_dirs)}
    samples: list[tuple[str, int]] = []

    for directory in class_dirs:
        files = sorted(
            path for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        samples.extend((str(path), class_indices[directory.name]) for path in files)

    if not samples:
        raise ValueError(f"No supported images found under {data_dir}")

    return samples, class_indices


def split_samples(
    samples: list[tuple[str, int]], validation_fraction: float, seed: int
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Create a deterministic train/validation split."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    shuffled = samples.copy()
    random.Random(seed).shuffle(shuffled)
    split_index = int(len(shuffled) * (1 - validation_fraction))
    if split_index == 0 or split_index == len(shuffled):
        raise ValueError("Dataset is too small for the requested validation split")
    return shuffled[:split_index], shuffled[split_index:]


def make_dataset(
    samples: list[tuple[str, int]],
    image_size: tuple[int, int],
    batch_size: int,
    num_classes: int,
    training: bool,
) -> tf.data.Dataset:
    """Build a tf.data pipeline without deleting or modifying source images."""
    paths = [path for path, _ in samples]
    labels = [label for _, label in samples]

    def load(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0

        if training:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.15)
            image = tf.image.random_contrast(image, 0.8, 1.2)

        return image, tf.one_hot(label, num_classes)

    return (
        tf.data.Dataset.from_tensor_slices((paths, labels))
        .map(load, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def class_weights(samples: list[tuple[str, int]]) -> dict[int, float]:
    """Compute balanced class weights from training labels."""
    labels = np.asarray([label for _, label in samples])
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def build_model(num_classes: int, image_size: tuple[int, int]) -> Model:
    """Build an EfficientNetB0 transfer-learning classifier."""
    inputs = tf.keras.Input(shape=(*image_size, 3))
    backbone = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)
    backbone.trainable = False

    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return Model(inputs, outputs, name="pcos_ultrasound_efficientnet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")

    set_seed(args.seed)
    samples, class_indices = discover_samples(args.data_dir)
    train_samples, val_samples = split_samples(samples, args.validation_fraction, args.seed)
    image_size = (224, 224)
    num_classes = len(class_indices)

    train_ds = make_dataset(train_samples, image_size, args.batch_size, num_classes, True)
    val_ds = make_dataset(val_samples, image_size, args.batch_size, num_classes, False)

    model = build_model(num_classes, image_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc", multi_label=True)],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            args.output_dir / "best_model.keras", monitor="val_auc", mode="max", save_best_only=True
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights(train_samples),
        callbacks=callbacks,
    )

    with open(args.output_dir / "class_indices.json", "w", encoding="utf-8") as handle:
        json.dump(class_indices, handle, indent=2)

    model.save(args.output_dir / "final_model.keras")
    with open(args.output_dir / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump({"classes": class_indices, "train_samples": len(train_samples), "validation_samples": len(val_samples), "history": history.history}, handle, indent=2)


if __name__ == "__main__":
    main()
