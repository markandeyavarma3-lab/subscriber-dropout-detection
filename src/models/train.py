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
from datetime import date, datetime, timedelta, timezone
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
    # Populated only when --promote ran. An orchestrator needs the verdict as
    # data, not as a log line it would have to scrape.
    promotion: dict[str, Any] | None = None
    mlflow_run_id: str | None = None
    # Calibration, cost and fairness. Carried on the result rather than only
    # written to metrics.json, so the orchestrator can escalate on a fairness
    # failure instead of having to re-read and re-parse the artifact.
    decision_quality: dict[str, Any] | None = None


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


def _load_splits(
    source: str,
    data_path: Path | None,
    n_subscribers: int | None,
    test_size: float,
    validation_size: float,
    seed: int,
    cutoffs: list[str] | None,
    log,
) -> tuple[DataSplits, dict[str, Any]]:
    """Build train/validation/test splits from the chosen data source.

    ``warehouse`` splits by *time* - fit on earlier cutoffs, score later ones -
    which is the only honest way to validate a churn model.  ``csv`` keeps the
    original random stratified split so the legacy path, CI and the existing
    tests continue to work unchanged.
    """
    if source == "warehouse":
        from src.features.point_in_time import build_temporal_splits, monthly_cutoffs
        from src.warehouse.simulate import ensure_warehouse

        populated = ensure_warehouse(
            n_subscribers=n_subscribers, seed=seed
        )
        if populated is not None:
            log(f"Warehouse was empty; simulated it -> {populated.summary()}")

        chosen = cutoffs or [
            cutoff.isoformat()
            for cutoff in monthly_cutoffs(
                settings.SIMULATION_START,
                # Leave room at the end for the label horizon, or the final
                # cutoff's labels would be censored by the simulation ending.
                date.fromisoformat(settings.SIMULATION_END)
                - timedelta(days=settings.PREDICTION_HORIZON_DAYS),
            )
        ]
        splits, assignment = build_temporal_splits(chosen)
        log(f"Temporal split over {len(chosen)} cutoffs -> {assignment}")
        return splits, {"source": "warehouse", "cutoffs": assignment}

    dataset_path = ensure_dataset(data_path, n_subscribers=n_subscribers)
    frame = load_raw_data(dataset_path)
    log(f"Loaded {len(frame):,} subscribers from {dataset_path}")
    log(f"Dropout rate: {frame[settings.TARGET_COLUMN].mean():.1%}")
    splits = split_data(
        frame, test_size=test_size, validation_size=validation_size, seed=seed
    )
    return splits, {
        "source": "csv",
        "path": str(dataset_path),
        "rows": int(len(frame)),
        "dropout_rate": round(float(frame[settings.TARGET_COLUMN].mean()), 4),
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
    source: str = "csv",
    cutoffs: list[str] | None = None,
    track: bool = False,
    promote_model: bool = False,
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

    splits, dataset_info = _load_splits(
        source, data_path, n_subscribers, test_size, validation_size, seed, cutoffs, log
    )
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
    decision_report = _decision_quality(model, splits, threshold, log)

    model_path = output_dir / "model.joblib"
    metrics_path = output_dir / "metrics.json"
    metadata_path = output_dir / "metadata.json"
    profile_path = output_dir / "reference_profile.json"

    joblib.dump(model, model_path)
    metrics_report = {
        "model_name": settings.MODEL_NAME,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision_threshold": threshold,
        "dataset": {**dataset_info, **splits.summary()},
        "validation": validation_metrics,
        "test": test_metrics,
        "top_feature_importances": importances,
        # Accuracy metrics say how well the model ranks. These say whether it
        # is fit for the decision it drives: are the probabilities meaningful,
        # is the threshold worth money, does it work for everyone.
        "decision_quality": decision_report,
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

    promotion_outcome: dict[str, Any] | None = None
    mlflow_run_id: str | None = None
    if track:
        mlflow_run_id, promotion_outcome = _track_and_maybe_promote(
            model=model,
            resolved_params=resolved_params,
            threshold=threshold,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            dataset_info=dataset_info,
            splits=splits,
            promote_model=promote_model,
            log=log,
        )

    return TrainingResult(
        model=model,
        threshold=threshold,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        model_path=model_path,
        metrics_path=metrics_path,
        metadata_path=metadata_path,
        reference_profile_path=profile_path,
        promotion=promotion_outcome,
        mlflow_run_id=mlflow_run_id,
        decision_quality=decision_report,
    )


def _decision_quality(
    model: Pipeline, splits: DataSplits, threshold: float, log
) -> dict[str, Any]:
    """Calibration, cost and fairness for the trained model.

    Reported alongside the accuracy metrics rather than instead of them: a
    model can rank well and still produce probabilities that mean nothing, a
    threshold that loses money, and predictions that work far better for some
    people than others.

    Never raises. These are diagnostics; a failure here must not cost a
    training run that has already produced a usable model.
    """
    from src.evaluation.fairness import disparity_report
    from src.models.calibration import calibration_metrics, compare_calibration
    from src.models.costs import CostModel, threshold_report

    try:
        proba = model.predict_proba(splits.X_test)[:, 1]
        report: dict[str, Any] = {
            "calibration": calibration_metrics(splits.y_test, proba),
            "costs": threshold_report(splits.y_test, proba, threshold, CostModel()),
        }

        # Calibration is only applied if it actually helps - isotonic can
        # overfit a small validation split and make the probabilities worse.
        from src.models.calibration import calibrate_pipeline

        try:
            calibrated = calibrate_pipeline(
                model, splits.X_val, splits.y_val, method=settings.CALIBRATION_METHOD
            )
            comparison = compare_calibration(
                splits.y_test, proba, calibrated.predict_proba(splits.X_test)[:, 1]
            )
            report["calibration_attempt"] = {
                "method": settings.CALIBRATION_METHOD,
                "improved": comparison["improved"],
                "ece_improvement": comparison["ece_improvement"],
                "applied": False,
            }
            log(
                f"Calibration ({settings.CALIBRATION_METHOD}): "
                + ("improved" if comparison["improved"] else "did NOT improve")
                + f" ECE by {comparison['ece_improvement']:+.5f} - not applied"
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics must not fail training
            report["calibration_attempt"] = {"error": str(exc), "applied": False}
            log(f"Calibration comparison failed: {exc}")

        # Slice by plan_type, which the model *does* see. acquisition_channel -
        # which it never sees - is the more interesting slice, but it is not
        # carried on the feature frame; see the fairness section of the README.
        if "plan_type" in splits.X_test.columns:
            report["fairness"] = disparity_report(
                splits.y_test, proba, splits.X_test["plan_type"], threshold, attribute="plan_type"
            )
            if not report["fairness"]["passes"]:
                log(f"Fairness concerns: {report['fairness']['concerns']}")

        savings = report["costs"]["savings"]
        log(
            f"Cost-optimal threshold {report['costs']['cost_optimal']['threshold']} "
            f"vs {threshold} in use -> {savings:,.0f} on the test split"
        )

        # Those savings assume every flagged subscriber gets contacted. Say out
        # loud how many that is, because it is the number an operations team
        # will push back on first.
        capacity = report["costs"]["capacity"]
        wanted = capacity["unconstrained"]["offers_required"]
        if capacity["binding"]:
            log(
                f"Capacity binds: {capacity['max_offers']} offers available against "
                f"{wanted} the model would send. The shortfall costs "
                f"{capacity['cost_of_constraint']:,.0f} on this split."
            )
        else:
            log(
                f"Cost-optimal outreach needs {wanted} offers "
                f"({wanted / max(capacity['population'], 1):.1%} of the split); "
                "no capacity cap configured."
            )
        return report
    except Exception as exc:  # noqa: BLE001 - never lose a good model to a diagnostic
        return {"error": str(exc)}


def _track_and_maybe_promote(
    model: Pipeline,
    resolved_params: dict[str, Any],
    threshold: float,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    dataset_info: dict[str, Any],
    splits: DataSplits,
    promote_model: bool,
    log,
) -> tuple[str | None, dict[str, Any] | None]:
    """Log the run to MLflow and, if asked, run it through the promotion gate.

    Imported lazily so the core training path keeps working - and CI keeps
    passing - on an install without MLflow.

    Returns:
        ``(run_id, promotion_decision)``; the decision is ``None`` when
        promotion was not requested.
    """
    from src.registry import promote as promotion
    from src.registry import tracking

    run_id, version = tracking.log_training_run(
        model=model,
        params={**resolved_params, "decision_threshold": threshold, **_flat_dataset(dataset_info)},
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        training_window=None,
        input_example=splits.X_train.head(5),
    )
    log(f"Logged MLflow run     -> {run_id}")
    if version is not None:
        log(f"Registered version    -> {version.version} (@challenger)")

    if not promote_model:
        return run_id, None

    # The gate is scored on the test split: data the challenger did not train
    # on, and which the champion has never seen either.
    decision = promotion.evaluate_promotion(
        challenger=model,
        features=splits.X_test,
        target=splits.y_test,
        challenger_version=version.version if version else None,
    )
    promotion.promote(decision)
    log(decision.summary())
    return run_id, decision.to_dict()


def _flat_dataset(dataset_info: dict[str, Any]) -> dict[str, Any]:
    """Flatten dataset provenance into MLflow-loggable scalar params."""
    flat: dict[str, Any] = {}
    for key, value in dataset_info.items():
        flat[f"data_{key}"] = str(value) if isinstance(value, dict) else value
    return flat


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
    parser.add_argument(
        "--source",
        choices=("csv", "warehouse"),
        default="csv",
        help="csv: legacy flat file with a random split. warehouse: "
        "point-in-time features from the event log, split by time.",
    )
    parser.add_argument(
        "--cutoffs",
        nargs="+",
        default=None,
        help="Explicit cutoff dates for --source warehouse (default: every 30 days).",
    )
    parser.add_argument(
        "--track", action="store_true", help="Log the run to MLflow and register the model."
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Run the challenger through the promotion gate against the champion.",
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
        source=args.source,
        cutoffs=args.cutoffs,
        # Promotion implies tracking: an unregistered model cannot be promoted.
        track=args.track or args.promote,
        promote_model=args.promote,
    )


if __name__ == "__main__":
    main()
