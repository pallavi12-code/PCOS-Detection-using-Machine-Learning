"""Train a reproducible clinical tabular baseline from a CSV file.

The CSV must contain a binary target column. Non-numeric values are coerced to
missing values and imputed using statistics learned from the training split.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def clean_numeric(value: object) -> float:
    """Extract the first numeric value, returning NaN when none is present."""
    if pd.isna(value):
        return float("nan")
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else float("nan")


def sanitize_column(name: str) -> str:
    """Create a stable feature name from a source column name."""
    return re.sub(r"_+", "_", re.sub(r"[^\w]", "_", name.strip())).strip("_")


def train_baseline(
    data: pd.DataFrame, target_column: str, seed: int
) -> tuple[Pipeline, dict[str, object]]:
    if target_column not in data.columns:
        raise KeyError(f"Target column not found: {target_column}")
    frame = data.drop(columns=["Sl. No", "Patient File No.", "Unnamed: 44"], errors="ignore")
    frame = frame.rename(columns={column: sanitize_column(column) for column in frame.columns})
    target = sanitize_column(target_column)
    if target not in frame.columns:
        raise KeyError(f"Target column not found after sanitizing: {target}")
    frame = frame.apply(lambda column: column.map(clean_numeric))
    X = frame.drop(columns=[target])
    y = frame[target].astype(int)
    if y.nunique() != 2:
        raise ValueError("The target column must contain exactly two classes")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
            )),
        ]
    )
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, probabilities >= 0.5)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "feature_columns": list(X.columns),
    }
    return pipeline, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/clinical_model.joblib"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model, metrics = train_baseline(pd.read_csv(args.data), args.target_column, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, **metrics}, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
