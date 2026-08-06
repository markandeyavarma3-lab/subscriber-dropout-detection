"""In-memory statistics about the predictions this process has served.

The point of tracking scores at serving time is that input drift and output
drift fail differently: a model can keep receiving perfectly ordinary-looking
inputs while its score distribution slides, and the first sign is usually that
the flagged rate moves without anyone changing anything.

Deliberately a bounded, in-process window - not a metrics backend.  It answers
"what is this replica doing right now" for a single deployable, and resets when
the process does.  Anything durable or cross-replica belongs in Prometheus or a
warehouse; see the future-work notes in the README.
"""

from __future__ import annotations

import threading
from collections import Counter, deque
from typing import Any

import numpy as np

from src.config import settings


class PredictionTracker:
    """A bounded rolling window of served prediction scores."""

    def __init__(self, window: int | None = None) -> None:
        self._window = window or settings.METRICS_WINDOW
        self._probabilities: deque[float] = deque(maxlen=self._window)
        self._labels: deque[int] = deque(maxlen=self._window)
        self._risk_levels: deque[str] = deque(maxlen=self._window)
        # Survives the rolling window, so the totals are not silently capped.
        self._served_total = 0
        self._lock = threading.Lock()

    def record(self, probability: float, label: int, risk_level: str) -> None:
        """Record one served prediction."""
        with self._lock:
            self._probabilities.append(float(probability))
            self._labels.append(int(label))
            self._risk_levels.append(str(risk_level))
            self._served_total += 1

    def record_many(self, responses: list[dict[str, Any]]) -> None:
        """Record a batch of prediction response payloads."""
        for response in responses:
            self.record(
                response["dropout_probability"],
                response["predicted_label"],
                response["risk_level"],
            )

    def reset(self) -> None:
        """Clear the window and the running total."""
        with self._lock:
            self._probabilities.clear()
            self._labels.clear()
            self._risk_levels.clear()
            self._served_total = 0

    @property
    def probabilities(self) -> list[float]:
        """A snapshot copy of the scores currently in the window."""
        with self._lock:
            return list(self._probabilities)

    def snapshot(self) -> dict[str, Any]:
        """Summarise the current window.

        Returns zeroed counters rather than nulls when nothing has been served,
        so a dashboard can render the payload without special-casing an empty
        service.
        """
        with self._lock:
            scores = np.array(self._probabilities, dtype=float)
            labels = list(self._labels)
            levels = Counter(self._risk_levels)
            served_total = self._served_total

        if scores.size == 0:
            return {
                "served_total": served_total,
                "window_size": 0,
                "window_capacity": self._window,
                "flagged_rate": 0.0,
                "probability": {"mean": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0},
                "risk_levels": {"low": 0, "medium": 0, "high": 0},
            }

        return {
            "served_total": served_total,
            "window_size": int(scores.size),
            "window_capacity": self._window,
            "flagged_rate": round(float(np.mean(labels)), 4),
            "probability": {
                "mean": round(float(scores.mean()), 4),
                "p50": round(float(np.percentile(scores, 50)), 4),
                "p90": round(float(np.percentile(scores, 90)), 4),
                "min": round(float(scores.min()), 4),
                "max": round(float(scores.max()), 4),
            },
            "risk_levels": {
                level: int(levels.get(level, 0)) for level in ("low", "medium", "high")
            },
        }


# Process-wide tracker used by the API.
_tracker = PredictionTracker()


def get_tracker() -> PredictionTracker:
    """Return the process-wide prediction tracker."""
    return _tracker
