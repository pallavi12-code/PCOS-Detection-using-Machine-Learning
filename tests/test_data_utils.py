from pathlib import Path

import pytest

from data_utils import discover_samples, split_samples


def _dataset(tmp_path: Path) -> Path:
    for class_name in ("normal", "pcos"):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for index in range(3):
            (class_dir / f"{index}.jpg").write_bytes(b"image")
        (class_dir / "notes.txt").write_text("ignored", encoding="utf-8")
    return tmp_path


def test_discover_samples_uses_sorted_classes_and_ignores_non_images(tmp_path):
    samples, class_indices = discover_samples(_dataset(tmp_path))

    assert class_indices == {"normal": 0, "pcos": 1}
    assert len(samples) == 6
    assert all(path.endswith(".jpg") for path, _ in samples)


def test_split_samples_is_deterministic_and_stratified(tmp_path):
    samples, _ = discover_samples(_dataset(tmp_path))

    train_a, validation_a = split_samples(samples, validation_fraction=0.33, seed=7)
    train_b, validation_b = split_samples(samples, validation_fraction=0.33, seed=7)

    assert train_a == train_b
    assert validation_a == validation_b
    assert {label for _, label in train_a} == {0, 1}
    assert {label for _, label in validation_a} == {0, 1}


def test_discover_samples_requires_multiple_classes(tmp_path):
    (tmp_path / "only").mkdir()
    (tmp_path / "only" / "image.jpg").write_bytes(b"image")

    with pytest.raises(ValueError, match="at least two"):
        discover_samples(tmp_path)
