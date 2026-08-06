"""Training entrypoint for the subscriber dropout model.

Run it either way::

    python -m src.models.train
    python src/models/train.py --n-estimators 150

The script loads (or generates) the dataset, fits a single scikit-learn
pipeline that owns both preprocessing and the classifier, tunes the decision
threshold on the validation split, and writes three artifacts:

``model.joblib``    the fitted pipeline, the only thing the API needs
``metrics.json``    validation and test metrics plus feature importances
``metadata.json``   decision threshold, feature list, library versions
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

# Allow `python src/models/train.py` by putting the project root on sys.path.
if __package__ in (None, ""):  # pragma: no cover - only hit via direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import settings  # noqa: E402
from src.data.loader import (  # noqa: E402
    DataSplits,
    ensure_dataset,
    load_raw_data,
    save_test_split,
    split_data,
)
from src.features.build_features import (  # noqa: E402
    REQUIRED_INPUT_COLUMNS,
    build_feature_pipeline,
)
from src.models.evaluate import (  # noqa: E402
    classification_text_report,
    evaluate_pipeline,
    format_metrics,
)
from src.monitoring.profile import (  # noqa: E402
    build_reference_profile,
    save_reference_profile,
)


@dataclass(frozen=True)
class TrainingResult:
    """Everything a caller (CLI or test) needs after a training run."""

    model: Pipeline
    threshold: float
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    model_path: Path
    metrics_path: Path
    metadata_path: Path
    reference_profile_path: Path


def build_model_pipeline(model_params: dict[str, Any] | None = None) -> Pipeline:
    """Assemble the end-to-end pipeline: features -> gradient boosting.

    Keeping preprocessing in the same object as the estimator means the saved
    artifact is self-contained and the API cannot apply a stale transformation.
    """
    params = {**settings.MODEL_PARAMS, **(model_params or {})}
    return Pipeline(
        steps=[
            ("features", build_feature_pipeline()),
            ("classifier", GradientBoostingClassifier(**params)),
        ]
    )


def tune_decision_threshold(
    y_true: pd.Series, y_proba: np.ndarray, grid: np.ndarray | None = None
) -> tuple[float, float]:
    """Pick the probability cut-off that maximises F1 on the validation split.

    A dropout model is used to trigger retention outreach, so the useful
    operating point is rarely the default 0.5 - especially with an imbalanced
    label.

    Returns:
        ``(threshold, best_f1)``.
    """
    from sklearn.metrics import f1_score

    # The grid is rounded *before* scoring: rounding the winner afterwards could
    # nudge the cut-off across a sample and leave the persisted threshold
    # disagreeing with the F1 reported for it.
    candidates = np.round(grid if grid is not None else np.arange(0.05, 0.96, 0.01), 3)
    best_threshold, best_score = settings.DECISION_THRESHOLD, -1.0
    for threshold in candidates:
        score = f1_score(y_true, (y_proba >= threshold).astype(int), zero_division=0)
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, round(best_score, 4)


def top_feature_importances(model: Pipeline, limit: int = 10) -> list[dict[str, float]]:
    """Return the most important features of a fitted pipeline, descending."""
    try:
        preprocessor = model.named_steps["features"].named_steps["preprocess"]
        names = list(preprocessor.get_feature_names_out())
        importances = model.named_steps["classifier"].feature_importances_
    except Exception:  # pragma: no cover - defensive across sklearn versions
        return []

    ranked = sorted(zip(names, importances, strict=True), key=lambda pair: pair[1], reverse=True)
    return [
        {"feature": name, "importance": round(float(value), 4)} for name, value in ranked[:limit]
    ]


def _build_metadata(
    model: Pipeline, threshold: float, splits: DataSplits, model_params: dict[str, Any]
) -> dict[str, Any]:
    """Assemble the metadata document saved next to the model."""
    return {
        "model_name": settings.MODEL_NAME,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision_threshold": threshold,
        "model_params": model_params,
        "required_input_columns": REQUIRED_INPUT_COLUMNS,
        "split_sizes": splits.summary(),
        "library_versions": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }


def run_training(
    data_path: Path | None = None,
    artifacts_dir: Path | None = None,
    model_params: dict[str, Any] | None = None,
    n_subscribers: int | None = None,
    test_size: float = settings.TEST_SIZE,
    validation_size: float = settings.VALIDATION_SIZE,
    seed: int = settings.RANDOM_SEED,
    tune_threshold: bool = settings.TUNE_THRESHOLD,
    persist_test_split: bool = True,
    test_split_path: Path | None = None,
    verbose: bool = True,
) -> TrainingResult:
    """Run the full training pipeline and persist the artifacts.

    Args:
        data_path: Raw CSV to train on. Generated if it does not exist.
        artifacts_dir: Directory for ``model.joblib`` and the JSON reports.
        model_params: Overrides merged over ``settings.MODEL_PARAMS``.
        n_subscribers: Row count used only when generating a missing dataset.
        test_size: Fraction of the data held out as the test split.
        validation_size: Fraction held out for threshold tuning and reporting.
        seed: Random seed for splitting.
        tune_threshold: Whether to search for the best-F1 threshold.
        persist_test_split: Whether to write the test split for ``evaluate.py``.
        test_split_path: Where to write that split. Defaults to the configured
            processed-data path, or to ``artifacts_dir`` when one is given, so a
            redirected run keeps all of its outputs together instead of
            overwriting the project's default split.
        verbose: Whether to print progress and metrics.

    Returns:
        A :class:`TrainingResult` with the fitted model, metrics and paths.
    """
    output_dir = artifacts_dir or settings.ARTIFACTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        if verbose:
            print(message)

    dataset_path = ensure_dataset(data_path, n_subscribers=n_subscribers)
    frame = load_raw_data(dataset_path)
    log(f"Loaded {len(frame):,} subscribers from {dataset_path}")
    log(f"Dropout rate: {frame[settings.TARGET_COLUMN].mean():.1%}")

    splits = split_data(frame, test_size=test_size, validation_size=validation_size, seed=seed)
    log(
        "Split -> train={train_rows:,} validation={validation_rows:,} "
        "test={test_rows:,}".format(**splits.summary())
    )

    resolved_params = {**settings.MODEL_PARAMS, **(model_params or {})}
    model = build_model_pipeline(resolved_params)
    log(f"Training {settings.MODEL_NAME} with {resolved_params}")
    model.fit(splits.X_train, splits.y_train)

    val_proba = model.predict_proba(splits.X_val)[:, 1]
    if tune_threshold:
        threshold, best_f1 = tune_decision_threshold(splits.y_val, val_proba)
        log(f"Tuned decision threshold: {threshold:.2f} (validation F1={best_f1:.4f})")
    else:
        threshold = settings.DECISION_THRESHOLD

    validation_metrics = evaluate_pipeline(model, splits.X_val, splits.y_val, threshold=threshold)
    test_metrics = evaluate_pipeline(model, splits.X_test, splits.y_test, threshold=threshold)

    if verbose:
        print()
        print(format_metrics("Validation", validation_metrics))
        print(format_metrics("Test", test_metrics))
        print()
        print(classification_text_report(splits.y_val, val_proba, threshold))

    importances = top_feature_importances(model)

    model_path = output_dir / "model.joblib"
    metrics_path = output_dir / "metrics.json"
    metadata_path = output_dir / "metadata.json"
    profile_path = output_dir / "reference_profile.json"

    joblib.dump(model, model_path)
    metrics_report = {
        "model_name": settings.MODEL_NAME,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision_threshold": threshold,
        "dataset": {
            "path": str(dataset_path),
            "rows": int(len(frame)),
            "dropout_rate": round(float(frame[settings.TARGET_COLUMN].mean()), 4),
            **splits.summary(),
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "top_feature_importances": importances,
    }
    metrics_path.write_text(json.dumps(metrics_report, indent=2))
    metadata_path.write_text(
        json.dumps(_build_metadata(model, threshold, splits, resolved_params), indent=2)
    )

    # The baseline is built from the *training* split specifically: it has to
    # describe the population the model actually learned from, not the whole
    # dataset including rows the model never saw.
    save_reference_profile(
        build_reference_profile(
            splits.X_train, probabilities=model.predict_proba(splits.X_train)[:, 1]
        ),
        profile_path,
    )

    if persist_test_split:
        destination = test_split_path
        if destination is None:
            destination = (
                artifacts_dir / "test.csv" if artifacts_dir else settings.TEST_DATA_PATH
            )
        log(f"Saved test split      -> {save_test_split(splits, destination)}")

    log(f"Saved model           -> {model_path}")
    log(f"Saved metrics report  -> {metrics_path}")
    log(f"Saved model metadata  -> {metadata_path}")
    log(f"Saved drift baseline  -> {profile_path}")

    return TrainingResult(
        model=model,
        threshold=threshold,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        model_path=model_path,
        metrics_path=metrics_path,
        metadata_path=metadata_path,
        reference_profile_path=profile_path,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train the subscriber dropout model.")
    parser.add_argument("--data-path", type=Path, default=None, help="Raw subscribers CSV.")
    parser.add_argument("--artifacts-dir", type=Path, default=None, help="Where to save artifacts.")
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED)
    parser.add_argument("--test-size", type=float, default=settings.TEST_SIZE)
    parser.add_argument("--validation-size", type=float, default=settings.VALIDATION_SIZE)
    parser.add_argument(
        "--no-tune-threshold",
        action="store_true",
        help="Use the configured threshold instead of tuning it on validation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> TrainingResult:
    """CLI entry point."""
    args = parse_args(argv)
    overrides = {
        key: value
        for key, value in {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "random_state": args.seed,
        }.items()
        if value is not None
    }
    return run_training(
        data_path=args.data_path,
        artifacts_dir=args.artifacts_dir,
        model_params=overrides,
        test_size=args.test_size,
        validation_size=args.validation_size,
        seed=args.seed,
        tune_threshold=not args.no_tune_threshold,
    )


if __name__ == "__main__":
    main()
