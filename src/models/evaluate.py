"""Metrics, reports, and standalone evaluation of a saved model artifact.

Used two ways:

* imported by :mod:`src.models.train` so training-time and evaluation-time
  metrics are computed by exactly the same code;
* run directly to verify that a persisted artifact still generalises::

      python -m src.models.evaluate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import settings


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    threshold: float = settings.DECISION_THRESHOLD,
) -> dict[str, Any]:
    """Compute the classification metric suite for one split.

    Args:
        y_true: Ground-truth labels (0/1).
        y_proba: Predicted probability of the positive (dropout) class.
        threshold: Probability cut-off for the hard label.

    Returns:
        A JSON-serialisable dict of metrics. ``roc_auc`` and ``pr_auc`` are
        ``None`` when the split contains a single class.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)

    both_classes_present = len(np.unique(y_true)) > 1
    metrics: dict[str, Any] = {
        "threshold": round(float(threshold), 4),
        "n_samples": int(len(y_true)),
        "positive_rate_actual": round(float(y_true.mean()), 4),
        "positive_rate_predicted": round(float(y_pred.mean()), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": (
            round(float(roc_auc_score(y_true, y_proba)), 4) if both_classes_present else None
        ),
        "pr_auc": (
            round(float(average_precision_score(y_true, y_proba)), 4)
            if both_classes_present
            else None
        ),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["confusion_matrix"] = {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    return metrics


def evaluate_pipeline(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = settings.DECISION_THRESHOLD,
) -> dict[str, Any]:
    """Score a fitted pipeline on one split and return its metrics."""
    y_proba = model.predict_proba(X)[:, 1]
    return compute_metrics(y, y_proba, threshold=threshold)


def format_metrics(name: str, metrics: dict[str, Any]) -> str:
    """Render a metrics dict as an aligned console block."""
    cm = metrics["confusion_matrix"]
    auc = metrics["roc_auc"]
    pr_auc = metrics["pr_auc"]
    lines = [
        f"--- {name} ({metrics['n_samples']:,} rows, threshold={metrics['threshold']:.2f}) ---",
        f"  accuracy   : {metrics['accuracy']:.4f}",
        f"  precision  : {metrics['precision']:.4f}",
        f"  recall     : {metrics['recall']:.4f}",
        f"  f1         : {metrics['f1']:.4f}",
        f"  roc_auc    : {auc:.4f}" if auc is not None else "  roc_auc    : n/a",
        f"  pr_auc     : {pr_auc:.4f}" if pr_auc is not None else "  pr_auc     : n/a",
        f"  confusion  : TN={cm['true_negatives']} FP={cm['false_positives']} "
        f"FN={cm['false_negatives']} TP={cm['true_positives']}",
    ]
    return "\n".join(lines)


def classification_text_report(
    y_true: np.ndarray | pd.Series, y_proba: np.ndarray, threshold: float
) -> str:
    """Return sklearn's per-class text report at the given threshold."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return classification_report(
        y_true, y_pred, target_names=["retained", "dropout"], zero_division=0
    )


def load_model(model_path: Path | None = None) -> Pipeline:
    """Load a serialized pipeline from disk."""
    import joblib

    path = model_path or settings.MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No model artifact at {path}. Train one with `python -m src.models.train`."
        )
    return joblib.load(path)


def load_threshold(metadata_path: Path | None = None) -> float:
    """Read the tuned decision threshold, falling back to the configured one."""
    path = metadata_path or settings.METADATA_PATH
    if path.exists():
        metadata = json.loads(path.read_text())
        return float(metadata.get("decision_threshold", settings.DECISION_THRESHOLD))
    return settings.DECISION_THRESHOLD


def main() -> dict[str, Any]:
    """CLI entry point: evaluate a saved artifact on the held-out test split."""
    parser = argparse.ArgumentParser(description="Evaluate a saved dropout model.")
    parser.add_argument("--model-path", type=Path, default=settings.MODEL_PATH)
    parser.add_argument("--test-path", type=Path, default=settings.TEST_DATA_PATH)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the decision threshold stored in metadata.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the evaluation report as JSON.",
    )
    args = parser.parse_args()

    if not args.test_path.exists():
        raise FileNotFoundError(
            f"No test split at {args.test_path}. Run `python -m src.models.train` first."
        )

    model = load_model(args.model_path)
    threshold = args.threshold if args.threshold is not None else load_threshold()

    frame = pd.read_csv(args.test_path)
    y_true = frame[settings.TARGET_COLUMN].astype(int)
    X = frame.drop(columns=[settings.TARGET_COLUMN, settings.ID_COLUMN], errors="ignore")

    y_proba = model.predict_proba(X)[:, 1]
    metrics = compute_metrics(y_true, y_proba, threshold=threshold)

    print(f"Model     : {args.model_path}")
    print(f"Test data : {args.test_path}")
    print(format_metrics("Held-out test set", metrics))
    print()
    print(classification_text_report(y_true, y_proba, threshold))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2))
        print(f"Wrote evaluation report to {args.output}")

    return metrics


if __name__ == "__main__":
    main()
