"""Shadow scoring: run the challenger on live traffic without serving it.

Every request is scored twice - once by ``@champion``, whose answer is
returned, and once by ``@challenger``, whose answer is only recorded. The
challenger sees exactly the traffic production sees, and no caller is ever
affected by it.

What this does and does not prove
---------------------------------

It is tempting to say shadow scoring shows which model is *better*. It does
not, and claiming otherwise is the most common mistake made with it. Shadow
traffic has **no labels**: nobody has churned yet, so there is no ground truth
to score either model against. Accuracy still has to come from the labelled
holdout the promotion gate uses (:mod:`src.registry.promote`).

What shadow scoring does prove is everything *else* that can go wrong on
deployment day:

- **Does the challenger survive production input?** Real traffic contains
  shapes no test fixture has - and a model that raises on 1% of live requests
  would have taken the service down had it been promoted.
- **What is the blast radius of switching?** An agreement rate of 97% means a
  promotion changes almost nothing. 60% means the outreach list is about to be
  rewritten, and someone should know that *before* it happens rather than from
  a surprised marketing team afterwards.
- **How much would flagging change?** If the challenger flags twice as many
  subscribers, retention costs double the day it is promoted, regardless of
  whether its AUC is higher.

So the two mechanisms answer different questions and are both required: the
gate asks "is it more accurate on data we have labels for", shadow asks "is it
safe to switch, and what changes when we do".
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.config import settings


@dataclass(frozen=True)
class ShadowComparison:
    """One request scored by both models."""

    champion_probability: float
    champion_label: int
    challenger_probability: float
    challenger_label: int

    @property
    def agreed(self) -> bool:
        """Whether both models would have taken the same action."""
        return self.champion_label == self.challenger_label

    @property
    def divergence(self) -> float:
        """Absolute gap between the two probabilities."""
        return abs(self.champion_probability - self.challenger_probability)


class ShadowTracker:
    """A bounded rolling window of champion/challenger comparisons.

    Bounded for the same reason the serving tracker is: memory has to stay
    flat under sustained traffic. The running totals survive the window, so
    "how many comparisons have we made" stays honest even after the samples
    themselves have rolled off.
    """

    def __init__(self, window: int | None = None) -> None:
        self._window = window or settings.SHADOW_WINDOW
        self._samples: deque[ShadowComparison] = deque(maxlen=self._window)
        self._compared_total = 0
        self._error_total = 0
        self._lock = threading.Lock()

    def record(self, comparison: ShadowComparison) -> None:
        """Record one paired scoring."""
        with self._lock:
            self._samples.append(comparison)
            self._compared_total += 1

    def record_error(self) -> None:
        """Record a challenger that failed to score.

        Counted rather than raised: a broken shadow model must never affect
        the response a caller receives. But it must not vanish either - a
        challenger erroring on live traffic is exactly the finding shadow
        scoring exists to surface before promotion.
        """
        with self._lock:
            self._error_total += 1

    def reset(self) -> None:
        """Clear the window and the running totals."""
        with self._lock:
            self._samples.clear()
            self._compared_total = 0
            self._error_total = 0

    def snapshot(self) -> dict[str, Any]:
        """Summarise the comparisons collected so far."""
        with self._lock:
            samples = list(self._samples)
            compared_total = self._compared_total
            error_total = self._error_total

        base: dict[str, Any] = {
            "compared_total": compared_total,
            "errors_total": error_total,
            "window_size": len(samples),
            "window_capacity": self._window,
        }

        if not samples:
            return {
                **base,
                "agreement_rate": None,
                "mean_absolute_divergence": None,
                "max_absolute_divergence": None,
                "champion_flagged_rate": None,
                "challenger_flagged_rate": None,
                "flagged_rate_delta": None,
                "challenger_flags_more": 0,
                "challenger_flags_fewer": 0,
            }

        champion_probs = np.array([s.champion_probability for s in samples])
        challenger_probs = np.array([s.challenger_probability for s in samples])
        champion_labels = np.array([s.champion_label for s in samples])
        challenger_labels = np.array([s.challenger_label for s in samples])

        divergence = np.abs(champion_probs - challenger_probs)
        champion_rate = float(champion_labels.mean())
        challenger_rate = float(challenger_labels.mean())

        return {
            **base,
            "agreement_rate": round(float((champion_labels == challenger_labels).mean()), 4),
            "mean_absolute_divergence": round(float(divergence.mean()), 4),
            "max_absolute_divergence": round(float(divergence.max()), 4),
            "champion_flagged_rate": round(champion_rate, 4),
            "challenger_flagged_rate": round(challenger_rate, 4),
            # The operational headline: how much the outreach list would change
            # in size the moment this challenger is promoted.
            "flagged_rate_delta": round(challenger_rate - champion_rate, 4),
            "challenger_flags_more": int(
                ((challenger_labels == 1) & (champion_labels == 0)).sum()
            ),
            "challenger_flags_fewer": int(
                ((challenger_labels == 0) & (champion_labels == 1)).sum()
            ),
        }

    def readiness(self) -> dict[str, Any]:
        """Whether enough traffic has accumulated to draw any conclusion.

        A 100% agreement rate over four requests means nothing. Reporting that
        as evidence would be worse than reporting nothing, because it looks
        like a result.
        """
        snapshot = self.snapshot()
        compared = snapshot["compared_total"]
        required = settings.SHADOW_MIN_COMPARISONS

        return {
            "sufficient_evidence": compared >= required,
            "compared_total": compared,
            "required": required,
        }


_tracker = ShadowTracker()


def get_shadow_tracker() -> ShadowTracker:
    """Return the process-wide shadow tracker."""
    return _tracker
