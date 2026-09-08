"""Dataset discovery and deterministic splitting helpers."""

from __future__ import annotations

import random
from pathlib import Path

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)
Sample = tuple[str, int]


def discover_samples(data_dir: Path) -> tuple[list[Sample], dict[str, int]]:
    """Return image paths and labels from ``data_dir/class_name/image``."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    class_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if len(class_dirs) < 2:
        raise ValueError("data_dir must contain at least two class directories")

    class_indices = {directory.name: index for index, directory in enumerate(class_dirs)}
    samples = [
        (str(path), class_indices[directory.name])
        for directory in class_dirs
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not samples:
        raise ValueError(f"No supported images found under {data_dir}")
    return samples, class_indices


def split_samples(
    samples: list[Sample], validation_fraction: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    """Split samples deterministically while preserving class proportions."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(samples) < 4:
        raise ValueError("At least four images are required for a split")

    by_class: dict[int, list[Sample]] = {}
    for sample in samples:
        by_class.setdefault(sample[1], []).append(sample)

    rng = random.Random(seed)
    train_samples: list[Sample] = []
    validation_samples: list[Sample] = []
    for label, class_samples in sorted(by_class.items()):
        if len(class_samples) < 2:
            raise ValueError("Each class must contain at least two images")
        shuffled = class_samples.copy()
        rng.shuffle(shuffled)
        validation_count = max(1, round(len(shuffled) * validation_fraction))
        if validation_count >= len(shuffled):
            validation_count = len(shuffled) - 1
        validation_samples.extend(shuffled[:validation_count])
        train_samples.extend(shuffled[validation_count:])

    rng.shuffle(train_samples)
    rng.shuffle(validation_samples)
    return train_samples, validation_samples
