"""Synthetic subscriber dataset generator.

The dataset is produced from a small generative story rather than independent
random columns, so the features are correlated the way they would be in a real
product database:

1.  Each subscriber gets three latent traits that are never written to the CSV:
    ``engagement``, ``dissatisfaction`` and ``price_sensitivity``.
2.  The observable columns are drawn conditionally on those traits and on the
    plan (a premium subscriber engages more, an unhappy one files more support
    tickets, a price-sensitive one hunts for discounts).
3.  The ``dropout`` label is drawn from a logistic model over the observable
    columns plus a noise term.  The noise is what keeps ROC-AUC in a believable
    0.85-0.92 band instead of a suspicious 1.0.

Run it with::

    python -m src.data.generate --n-subscribers 8000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import settings

PLAN_TYPES: tuple[str, ...] = ("basic", "standard", "premium")
PLAN_PROBABILITIES: tuple[float, ...] = (0.45, 0.35, 0.20)
PLAN_BASE_FEE: dict[str, float] = {"basic": 9.99, "standard": 19.99, "premium": 34.99}

# Baseline log-odds of dropout, calibrated so the generated base rate sits near
# the ~20% that a consumer subscription business typically sees.
DROPOUT_INTERCEPT: float = -1.80


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _standardise(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance transform used to keep logit weights readable."""
    std = x.std()
    return (x - x.mean()) / std if std > 0 else np.zeros_like(x)


def generate_subscribers(
    n_subscribers: int = settings.N_SUBSCRIBERS,
    seed: int = settings.RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a synthetic subscriber table with a ``dropout`` label.

    Args:
        n_subscribers: Number of rows to generate.
        seed: Seed for the random number generator, for reproducibility.

    Returns:
        A DataFrame with one row per subscriber.
    """
    rng = np.random.default_rng(seed)
    n = n_subscribers

    # --- Latent traits (not persisted) ------------------------------------ #
    engagement = rng.normal(0.0, 1.0, n)
    dissatisfaction = rng.normal(0.0, 1.0, n)
    price_sensitivity = rng.normal(0.0, 1.0, n)

    # --- Plan and pricing ------------------------------------------------- #
    plan_type = rng.choice(PLAN_TYPES, size=n, p=PLAN_PROBABILITIES)
    is_premium = (plan_type == "premium").astype(float)
    is_standard = (plan_type == "standard").astype(float)

    base_fee = np.array([PLAN_BASE_FEE[p] for p in plan_type])
    # Legacy/promotional pricing jitter, plus a small discount for the
    # price-sensitive cohort who negotiated or grabbed an offer.
    monthly_fee = base_fee + rng.normal(0.0, 1.6, n) - 0.9 * np.clip(price_sensitivity, 0, None)
    monthly_fee = np.round(np.clip(monthly_fee, 4.99, None), 2)

    # --- Tenure ----------------------------------------------------------- #
    tenure_days = np.clip(rng.lognormal(mean=5.3, sigma=0.9, size=n), 7, 2500).round()
    tenure_z = _standardise(np.log1p(tenure_days))

    # --- Auto renew: loyal, engaged and premium users opt in more --------- #
    auto_renew_logit = (
        0.35 + 0.55 * is_premium + 0.25 * is_standard + 0.50 * tenure_z + 0.40 * engagement
    )
    is_auto_renew_enabled = rng.random(n) < _sigmoid(auto_renew_logit)

    # --- Activity --------------------------------------------------------- #
    session_rate = np.exp(
        1.55 + 0.55 * engagement + 0.28 * is_premium + 0.12 * is_standard + 0.15 * tenure_z
    )
    session_counts = rng.poisson(session_rate)
    # Stored as an average, so a fractional part is expected.
    avg_session_count_last_30d = np.round(session_counts + rng.uniform(-0.4, 0.4, n), 1)
    avg_session_count_last_30d = np.clip(avg_session_count_last_30d, 0.0, None)

    # Recency is driven by activity: heavy users were seen very recently.
    recency_scale = 26.0 / (1.0 + 0.55 * session_counts)
    last_activity_days_ago = rng.exponential(recency_scale).round()
    # Cannot have been inactive for longer than the subscription has existed.
    last_activity_days_ago = np.minimum(last_activity_days_ago, tenure_days).astype(int)

    # --- Support and billing friction ------------------------------------- #
    ticket_rate = np.exp(-1.10 + 0.45 * dissatisfaction - 0.20 * engagement)
    support_tickets_last_90d = rng.poisson(ticket_rate)

    failure_rate = np.exp(
        -1.75
        + 0.45 * dissatisfaction
        - 0.85 * is_auto_renew_enabled.astype(float)
        + 0.20 * _standardise(monthly_fee)
    )
    payment_failures_last_6m = rng.poisson(failure_rate)

    discount_rate = np.exp(-0.95 + 0.40 * price_sensitivity - 0.25 * is_premium)
    discounts_used_last_6m = rng.poisson(discount_rate)

    # --- Dropout label ---------------------------------------------------- #
    # Weights are chosen so that each driver contributes a plausible amount of
    # log-odds over its realistic range, and so the total spread of the logit
    # lands the achievable ROC-AUC in a believable 0.85-0.90 band.  The noise
    # term plus the Bernoulli draw are what stop a model from reaching 1.0.
    dropout_logit = (
        DROPOUT_INTERCEPT
        + 0.115 * last_activity_days_ago
        - 0.180 * avg_session_count_last_30d
        - 1.40 * is_auto_renew_enabled.astype(float)
        + 1.10 * payment_failures_last_6m
        + 0.65 * support_tickets_last_90d
        + 0.30 * discounts_used_last_6m
        - 1.05 * tenure_z
        + 0.045 * (monthly_fee - 20.0)
        + rng.normal(0.0, 0.35, n)  # irreducible noise
    )
    dropout = (rng.random(n) < _sigmoid(dropout_logit)).astype(int)

    frame = pd.DataFrame(
        {
            settings.ID_COLUMN: [f"SUB-{i:06d}" for i in range(1, n + 1)],
            "tenure_days": tenure_days.astype(int),
            "plan_type": plan_type,
            "monthly_fee": monthly_fee,
            "avg_session_count_last_30d": avg_session_count_last_30d,
            "last_activity_days_ago": last_activity_days_ago,
            "support_tickets_last_90d": support_tickets_last_90d.astype(int),
            "payment_failures_last_6m": payment_failures_last_6m.astype(int),
            "discounts_used_last_6m": discounts_used_last_6m.astype(int),
            "is_auto_renew_enabled": is_auto_renew_enabled,
            settings.TARGET_COLUMN: dropout,
        }
    )
    return frame


def write_dataset(
    output_path: Path | None = None,
    n_subscribers: int = settings.N_SUBSCRIBERS,
    seed: int = settings.RANDOM_SEED,
) -> Path:
    """Generate the dataset and write it to CSV.

    Args:
        output_path: Destination CSV. Defaults to ``settings.RAW_DATA_PATH``.
        n_subscribers: Number of rows to generate.
        seed: Random seed.

    Returns:
        The path the CSV was written to.
    """
    destination = output_path or settings.RAW_DATA_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_subscribers(n_subscribers=n_subscribers, seed=seed)
    frame.to_csv(destination, index=False)
    return destination


def main() -> None:
    """CLI entry point: ``python -m src.data.generate``."""
    parser = argparse.ArgumentParser(description="Generate the synthetic subscriber dataset.")
    parser.add_argument("--n-subscribers", type=int, default=settings.N_SUBSCRIBERS)
    parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED)
    parser.add_argument("--output", type=Path, default=settings.RAW_DATA_PATH)
    args = parser.parse_args()

    path = write_dataset(output_path=args.output, n_subscribers=args.n_subscribers, seed=args.seed)
    frame = pd.read_csv(path)
    rate = frame[settings.TARGET_COLUMN].mean()
    print(f"Wrote {len(frame):,} subscribers to {path}")
    print(f"Dropout rate: {rate:.1%}")


if __name__ == "__main__":
    main()
