"""Evaluate a trained Keras image classifier on a class-folder dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from data_utils import discover_samples
from train import make_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    samples, class_indices = discover_samples(args.data_dir)
    model = tf.keras.models.load_model(args.model, compile=False)
    dataset = make_dataset(
        samples, (args.image_size, args.image_size), args.batch_size,
        len(class_indices), training=False, seed=0
    )
    metrics = model.evaluate(dataset, return_dict=True, verbose=1)
    print(json.dumps({key: float(value) for key, value in metrics.items()}, indent=2))


if __name__ == "__main__":
    main()
