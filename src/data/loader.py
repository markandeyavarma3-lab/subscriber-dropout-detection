"""Loading and splitting utilities for the subscriber dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import settings


@dataclass(frozen=True)
class DataSplits:
    """A stratified train / validation / test split of the subscriber table."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    def summary(self) -> dict[str, int]:
        """Row counts per split, handy for logging and metrics reports."""
        return {
            "train_rows": len(self.X_train),
            "validation_rows": len(self.X_val),
            "test_rows": len(self.X_test),
        }


def ensure_dataset(path: Path | None = None, n_subscribers: int | None = None) -> Path:
    """Return the raw dataset path, generating the CSV first if it is missing.

    This keeps a freshly cloned repository runnable with a single command: the
    training script calls it before loading.
    """
    destination = path or settings.RAW_DATA_PATH
    if not destination.exists():
        from src.data.generate import write_dataset

        write_dataset(
            output_path=destination,
            n_subscribers=n_subscribers or settings.N_SUBSCRIBERS,
        )
    return destination


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Read the subscriber CSV into a DataFrame.

    Args:
        path: CSV to read. Defaults to ``settings.RAW_DATA_PATH``.

    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    source = path or settings.RAW_DATA_PATH
    if not source.exists():
        raise FileNotFoundError(
            f"Dataset not found at {source}. Run `python -m src.data.generate` first."
        )
    frame = pd.read_csv(source)
    if settings.TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Dataset at {source} has no '{settings.TARGET_COLUMN}' column.")
    # pandas reads the boolean column back as bool already, but a CSV written by
    # another tool may use strings; normalise defensively.
    if frame["is_auto_renew_enabled"].dtype == object:
        frame["is_auto_renew_enabled"] = (
            frame["is_auto_renew_enabled"].astype(str).str.lower().isin({"true", "1", "yes"})
        )
    return frame


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into the model matrix and the target vector.

    The subscriber ID is dropped: it carries no signal and must never reach the
    model.
    """
    target = frame[settings.TARGET_COLUMN].astype(int)
    features = frame.drop(columns=[settings.TARGET_COLUMN, settings.ID_COLUMN], errors="ignore")
    return features, target


def split_data(
    frame: pd.DataFrame,
    test_size: float = settings.TEST_SIZE,
    validation_size: float = settings.VALIDATION_SIZE,
    seed: int = settings.RANDOM_SEED,
) -> DataSplits:
    """Split the dataset into stratified train / validation / test sets.

    ``test_size`` and ``validation_size`` are fractions of the *full* dataset;
    training receives whatever remains.

    Args:
        frame: Full subscriber table including the target column.
        test_size: Fraction held out as the final test set.
        validation_size: Fraction held out for model selection.
        seed: Seed controlling the shuffle.
    """
    if not 0 < test_size + validation_size < 1:
        raise ValueError("test_size + validation_size must be strictly between 0 and 1.")

    features, target = split_features_target(frame)

    X_rest, X_test, y_rest, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=seed,
        stratify=target,
    )

    # Re-express the validation fraction relative to the remaining rows.
    relative_val_size = validation_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest,
        y_rest,
        test_size=relative_val_size,
        random_state=seed,
        stratify=y_rest,
    )

    return DataSplits(
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )


def save_test_split(splits: DataSplits, path: Path | None = None) -> Path:
    """Persist the held-out test split so ``evaluate.py`` can reuse it verbatim."""
    destination = path or settings.TEST_DATA_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = splits.X_test.copy()
    frame[settings.TARGET_COLUMN] = splits.y_test.to_numpy()
    frame.to_csv(destination, index=False)
    return destination
