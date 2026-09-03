"""Group-wise performance: does this model work equally well for everyone?

An aggregate ROC-AUC is a weighted average, and averages hide their worst
cases. A model at 0.71 overall can be at 0.78 for long-tenured premium
subscribers and 0.55 - barely better than a coin - for people who arrived last
month through a paid ad. The aggregate never says so, and nobody finds out
until the retention budget has been quietly misallocated for a year.

Two kinds of disparity, and they are not the same thing
------------------------------------------------------

**Selection disparity** - some groups get flagged far more often than others.
This is often *correct*: if one cohort genuinely churns more, flagging them
more is the model working. It is a finding to interpret, not automatically a
fault.

**Performance disparity** - the model is measurably *worse* for some group, so
its predictions there are less trustworthy. This is much harder to defend. It
means the people in that group receive worse decisions than everybody else,
whichever direction the error runs.

Reporting only the first is the common mistake: it produces an alarming table
about groups that simply differ, while a group the model genuinely cannot
predict passes unnoticed.

On thresholds
-------------

The 0.8 disparity ratio here comes from the US "four-fifths rule". It is a
convention borrowed to have *some* agreed starting point, not a legal standard
and not a law of nature - a ratio of 0.79 is not a violation and 0.81 is not
a clean bill of health. Treat a flag as "go and look", never as a verdict.

What this cannot do
-------------------

The data is simulated, so any disparity found here is a property of the
generator, not evidence about real subscribers. The machinery is real and would
work on real data; the specific numbers are not findings about the world.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import settings

logger = logging.getLogger(__name__)


def group_metrics(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    groups: np.ndarray | pd.Series,
    threshold: float,
    min_size: int | None = None,
) -> pd.DataFrame:
    """Per-group performance and selection rates.

    Args:
        y_true: Ground-truth labels.
        y_proba: Predicted probabilities.
        groups: Group label per row - the attribute to slice by.
        threshold: Decision threshold applied to produce hard labels.
        min_size: Below this, a group's metrics are marked untrustworthy
            rather than dropped. Dropping small groups silently is how a
            problem affecting a minority disappears from the report.

    Returns:
        One row per group, ordered by size descending.
    """
    floor = min_size if min_size is not None else settings.FAIRNESS_MIN_GROUP_SIZE

    truth = np.asarray(y_true).astype(int)
    proba = np.asarray(y_proba, dtype=float)
    labels = pd.Series(np.asarray(groups)).astype(str).to_numpy()
    predicted = (proba >= threshold).astype(int)

    rows: list[dict[str, Any]] = []
    for name in pd.unique(labels):
        mask = labels == name
        size = int(mask.sum())
        group_truth = truth[mask]
        group_pred = predicted[mask]

        positives = int(group_truth.sum())
        true_positives = int(((group_pred == 1) & (group_truth == 1)).sum())
        false_positives = int(((group_pred == 1) & (group_truth == 0)).sum())
        false_negatives = int(((group_pred == 0) & (group_truth == 1)).sum())

        # AUC needs both classes present; a group that never churns has no
        # ranking to score, which is missing information rather than zero.
        both_classes = len(np.unique(group_truth)) > 1
        auc = float(roc_auc_score(group_truth, proba[mask])) if both_classes else None

        rows.append(
            {
                "group": name,
                "n": size,
                "base_rate": round(float(group_truth.mean()), 4) if size else 0.0,
                # Selection rate: how often this group gets flagged at all.
                "selection_rate": round(float(group_pred.mean()), 4) if size else 0.0,
                "recall": (
                    round(true_positives / positives, 4) if positives else None
                ),
                "precision": (
                    round(true_positives / (true_positives + false_positives), 4)
                    if (true_positives + false_positives)
                    else None
                ),
                "false_negative_rate": (
                    round(false_negatives / positives, 4) if positives else None
                ),
                "roc_auc": round(auc, 4) if auc is not None else None,
                "sufficient_sample": size >= floor,
            }
        )

    table = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    # pandas coerces None to NaN in float columns, and NaN is not valid JSON -
    # a report serialised with it would fail the moment it reached an artifact
    # or an endpoint. Keep the column object-typed so "undefined" stays None.
    for column in ("roc_auc", "recall", "precision", "false_negative_rate"):
        table[column] = table[column].astype(object).where(table[column].notna(), None)
    return table


def _ratio(values: list[float]) -> float | None:
    """Worst-to-best ratio, the four-fifths-rule shape.

    Filters NaN explicitly rather than relying on ``NaN > 0`` being False,
    which works but only by accident and breaks the moment the comparison is
    reordered.
    """
    usable = [v for v in values if v is not None and not pd.isna(v) and v > 0]
    if len(usable) < 2:
        return None
    return round(min(usable) / max(usable), 4)


def disparity_report(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    groups: np.ndarray | pd.Series,
    threshold: float,
    attribute: str = "group",
    min_size: int | None = None,
    disparity_threshold: float | None = None,
) -> dict[str, Any]:
    """Summarise disparity across one attribute.

    Distinguishes *selection* disparity (some groups flagged more) from
    *performance* disparity (the model is worse for some groups), because they
    mean different things and only the second is straightforwardly a defect.

    Groups below ``min_size`` are excluded from the ratios - their metrics are
    noise - but still appear in ``groups`` so a small group is never silently
    dropped from the report.
    """
    limit = (
        disparity_threshold
        if disparity_threshold is not None
        else settings.FAIRNESS_DISPARITY_THRESHOLD
    )
    table = group_metrics(y_true, y_proba, groups, threshold, min_size)
    usable = table[table["sufficient_sample"]]

    selection_ratio = _ratio(list(usable["selection_rate"]))
    recall_ratio = _ratio(list(usable["recall"]))
    auc_ratio = _ratio(list(usable["roc_auc"]))

    concerns: list[str] = []
    if selection_ratio is not None and selection_ratio < limit:
        concerns.append(
            f"selection rates differ by more than the {limit:g} ratio "
            f"({selection_ratio:g}) - check whether the underlying churn rates "
            "differ too, which would make this expected"
        )
    if recall_ratio is not None and recall_ratio < limit:
        concerns.append(
            f"recall differs by more than the {limit:g} ratio ({recall_ratio:g}) - "
            "some groups' churners are being missed far more often"
        )
    if auc_ratio is not None and auc_ratio < limit:
        concerns.append(
            f"discrimination differs by more than the {limit:g} ratio ({auc_ratio:g}) - "
            "the model is measurably less able to rank some groups"
        )

    ranked = usable[usable["roc_auc"].notna()].copy()
    ranked["roc_auc"] = ranked["roc_auc"].astype(float)
    worst_auc = ranked.nsmallest(1, "roc_auc")

    return {
        "attribute": attribute,
        "threshold": round(float(threshold), 4),
        "n_groups": int(len(table)),
        "n_groups_evaluated": int(len(usable)),
        "disparity_threshold": limit,
        "selection_rate_ratio": selection_ratio,
        "recall_ratio": recall_ratio,
        "roc_auc_ratio": auc_ratio,
        "weakest_group": (
            {
                "group": worst_auc.iloc[0]["group"],
                "roc_auc": worst_auc.iloc[0]["roc_auc"],
                "n": int(worst_auc.iloc[0]["n"]),
            }
            if not worst_auc.empty
            else None
        ),
        "concerns": concerns,
        "passes": not concerns,
        "groups": table.to_dict("records"),
    }


def audit(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    attributes: dict[str, np.ndarray | pd.Series],
    threshold: float,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a disparity report across several attributes at once.

    Returns:
        A report per attribute plus an overall ``passes`` flag, so a pipeline
        can gate or alert on the aggregate without losing the detail.
    """
    reports = {
        name: disparity_report(y_true, y_proba, values, threshold, attribute=name, **kwargs)
        for name, values in attributes.items()
    }

    return {
        "passes": all(report["passes"] for report in reports.values()),
        "attributes_with_concerns": [
            name for name, report in reports.items() if not report["passes"]
        ],
        "reports": reports,
    }
