"""Build the reference distribution snapshot used as the drift baseline.

A trained model only makes sense on data that looks like what it was trained
on.  This module captures what "looks like" meant at training time: for every
raw input column, the shape of its distribution, plus the shape of the model's
own output.  :mod:`src.monitoring.drift` later scores live traffic against it.

Distributions are stored as **quantile bins** rather than as raw rows.  That
keeps the artifact small and constant-size, avoids shipping a copy of the
training data next to the model, and is exactly the form the Population
Stability Index needs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import settings
from src.features.build_features import (
    CATEGORICAL_FEATURES,
    RAW_BINARY_FEATURES,
    RAW_NUMERIC_FEATURES,
)

# Columns treated as discrete: the plan tier plus the boolean flags.
DISCRETE_FEATURES: list[str] = CATEGORICAL_FEATURES + RAW_BINARY_FEATURES


def _clean(values: pd.Series) -> np.ndarray:
    """Return the finite numeric values of a column as a float array."""
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return numeric[np.isfinite(numeric)]


def numeric_bin_edges(values: pd.Series, n_bins: int | None = None) -> list[float]:
    """Return quantile bin edges describing a numeric column.

    Quantile bins are used rather than equal-width ones so that every bin
    carries a comparable share of the reference population; equal-width bins on
    a skewed column leave most bins nearly empty, which makes PSI explode on
    tiny, meaningless movements.

    Duplicate edges are collapsed, so a heavily tied column simply ends up with
    fewer bins instead of zero-width ones.
    """
    bins = n_bins or settings.DRIFT_BIN_COUNT
    clean = _clean(values)
    if clean.size == 0:
        return [0.0, 1.0]

    edges = np.unique(np.quantile(clean, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        # A constant column: manufacture a unit-wide bin around the value so
        # downstream histogramming still has somewhere to put the data.
        centre = float(edges[0])
        return [centre - 0.5, centre + 0.5]
    return [float(edge) for edge in edges]


def proportions_in_bins(values: pd.Series, edges: list[float]) -> list[float]:
    """Share of ``values`` falling in each bin defined by ``edges``.

    Values beyond the reference range are clipped into the outer bins: a live
    value larger than anything seen in training is still evidence about the top
    of the distribution, not something to discard.
    """
    clean = _clean(values)
    if clean.size == 0:
        return [0.0] * (len(edges) - 1)

    counts, _ = np.histogram(np.clip(clean, edges[0], edges[-1]), bins=edges)
    total = counts.sum()
    if total == 0:  # pragma: no cover - only if every value were non-finite
        return [0.0] * len(counts)
    return [float(count / total) for count in counts]


def category_proportions(values: pd.Series) -> dict[str, float]:
    """Relative frequency of each distinct value in a discrete column."""
    counts = values.astype(str).value_counts(normalize=True)
    return {str(category): float(share) for category, share in counts.items()}


def build_reference_profile(
    frame: pd.DataFrame, probabilities: np.ndarray | None = None
) -> dict[str, Any]:
    """Summarise a training frame into a JSON-serialisable reference profile.

    Args:
        frame: The raw training rows (before any feature engineering).
        probabilities: Model scores for those rows, so output drift can be
            detected even when the inputs look stable.

    Returns:
        A profile document ready to be written next to the model artifact.
    """
    numeric: dict[str, Any] = {}
    for column in RAW_NUMERIC_FEATURES:
        if column not in frame.columns:
            continue
        edges = numeric_bin_edges(frame[column])
        clean = _clean(frame[column])
        numeric[column] = {
            "edges": edges,
            "proportions": proportions_in_bins(frame[column], edges),
            "mean": round(float(clean.mean()), 4) if clean.size else 0.0,
            "std": round(float(clean.std()), 4) if clean.size else 0.0,
            "min": round(float(clean.min()), 4) if clean.size else 0.0,
            "max": round(float(clean.max()), 4) if clean.size else 0.0,
        }

    discrete = {
        column: {"proportions": category_proportions(frame[column])}
        for column in DISCRETE_FEATURES
        if column in frame.columns
    }

    profile: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_reference_rows": int(len(frame)),
        "bin_count": settings.DRIFT_BIN_COUNT,
        "numeric": numeric,
        "discrete": discrete,
    }

    if probabilities is not None and len(probabilities) > 0:
        scores = pd.Series(np.asarray(probabilities, dtype=float))
        # Fixed edges across [0, 1]: prediction drift is about where scores sit
        # on the probability scale, so the bins must not move between runs.
        edges = [round(edge, 4) for edge in np.linspace(0.0, 1.0, settings.DRIFT_BIN_COUNT + 1)]
        profile["prediction"] = {
            "edges": edges,
            "proportions": proportions_in_bins(scores, edges),
            "mean": round(float(scores.mean()), 4),
        }

    return profile


def save_reference_profile(profile: dict[str, Any], path: Path | None = None) -> Path:
    """Write a reference profile to disk, creating parent directories."""
    destination = path or settings.REFERENCE_PROFILE_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(profile, indent=2))
    return destination


def load_reference_profile(path: Path | None = None) -> dict[str, Any] | None:
    """Read a reference profile, returning ``None`` when there is not one.

    A missing or unreadable profile is not fatal: the service still predicts,
    it just cannot report drift.
    """
    source = path or settings.REFERENCE_PROFILE_PATH
    if not source.exists():
        return None
    try:
        return json.loads(source.read_text())
    except json.JSONDecodeError:  # pragma: no cover - corrupt artifact directory
        return None
