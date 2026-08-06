"""Generate a temporal event stream for the subscriber warehouse.

Unlike the flat generator this replaces, nothing here is pre-aggregated.  The
simulator walks a calendar day by day and emits the events a real service would
emit - sessions, payments, tickets, renewals, cancellations - leaving every
aggregate to be computed later, at a chosen cutoff, by SQL that can only look
backwards.

Each subscriber carries latent traits (``engagement``, ``dissatisfaction``,
``price_sensitivity``) that are never written to any table.  Observable
behaviour is drawn conditionally on those traits, which is what makes the
features genuinely predictive of the label without the label being a direct
function of any column.

**Injectable drift** is the point of building this rather than downloading a
static dataset.  A :class:`DriftScenario` changes subscriber behaviour from a
chosen date onward, so the monitoring, retraining and promotion machinery can
be demonstrated firing on real movement instead of described in a README.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np

from src.config import settings
from src.warehouse import schema
from src.warehouse.database import create_schema, insert_rows, truncate_all

logger = logging.getLogger(__name__)

PLAN_TYPES = ("basic", "standard", "premium")
PLAN_FEES = {"basic": 9.99, "standard": 19.99, "premium": 29.99}
PLAN_WEIGHTS = (0.45, 0.35, 0.20)
ACQUISITION_CHANNELS = ("organic", "paid_search", "referral", "social")
TICKET_CATEGORIES = ("billing", "technical", "content", "account")


@dataclass(frozen=True)
class DriftScenario:
    """A deliberate change in subscriber behaviour from ``starts_on`` onward.

    Defaults are all-neutral, so an unconfigured scenario changes nothing.

    Attributes:
        starts_on: Date the shift begins.
        engagement_multiplier: Scales session frequency (``0.5`` halves it).
        payment_failure_multiplier: Scales the per-payment failure rate.
        ticket_multiplier: Scales support ticket frequency.
        cancellation_multiplier: Scales the daily cancellation hazard.
        new_plan_share: Share of *new* signups placed on an unseen plan tier,
            which exercises the unknown-category path end to end.
        new_plan_name: Name of that unseen tier.
    """

    starts_on: date
    engagement_multiplier: float = 1.0
    payment_failure_multiplier: float = 1.0
    ticket_multiplier: float = 1.0
    cancellation_multiplier: float = 1.0
    new_plan_share: float = 0.0
    new_plan_name: str = "enterprise"

    def active_on(self, day: date) -> bool:
        """Whether the scenario applies on a given day."""
        return day >= self.starts_on


@dataclass
class _Subscriber:
    """Mutable simulation state for one subscriber. Never persisted directly."""

    subscriber_id: str
    signup_date: date
    plan_type: str
    monthly_fee: float
    auto_renew: bool
    engagement: float
    dissatisfaction: float
    price_sensitivity: float
    next_renewal: date
    active: bool = True
    cancelled_on: date | None = None
    failures: int = 0
    last_session_on: date | None = None


@dataclass
class SimulationResult:
    """What a simulation run produced."""

    counts: dict[str, int] = field(default_factory=dict)
    start: date | None = None
    end: date | None = None
    n_subscribers: int = 0

    def summary(self) -> str:
        """One-line human summary of the run."""
        rows = ", ".join(f"{name}={count:,}" for name, count in self.counts.items())
        return f"{self.start} -> {self.end}: {rows}"


def _to_date(value: str | date) -> date:
    """Coerce an ISO string or date into a date."""
    return value if isinstance(value, date) else date.fromisoformat(value)


def _at(day: date, rng: np.random.Generator) -> datetime:
    """Place an event at a random time of day, so timestamps are not all midnight."""
    return datetime.combine(day, datetime.min.time()) + timedelta(
        seconds=int(rng.integers(0, 86_400))
    )


def _make_subscriber(
    index: int, signup: date, rng: np.random.Generator, plan_pool: tuple[str, ...]
) -> _Subscriber:
    """Draw one subscriber's latent traits and starting plan."""
    plan = str(rng.choice(plan_pool, p=PLAN_WEIGHTS if len(plan_pool) == 3 else None))
    # Premium subscribers skew more engaged; this is what makes plan_type carry
    # signal rather than being noise.
    plan_lift = {"basic": -0.15, "standard": 0.0, "premium": 0.2}.get(plan, 0.0)

    return _Subscriber(
        subscriber_id=f"SUB-{index:07d}",
        signup_date=signup,
        plan_type=plan,
        monthly_fee=PLAN_FEES.get(plan, 49.99),
        auto_renew=bool(rng.random() < 0.72),
        engagement=float(np.clip(rng.beta(2.5, 2.5) + plan_lift, 0.02, 0.99)),
        dissatisfaction=float(rng.beta(2.0, 5.0)),
        price_sensitivity=float(rng.beta(2.0, 3.0)),
        next_renewal=signup + timedelta(days=30),
    )


def _daily_cancellation_hazard(
    sub: _Subscriber, scenario: DriftScenario | None, day: date
) -> float:
    """Probability this subscriber cancels today.

    Built from the same latent traits that drive observable behaviour, so the
    label is correlated with the features without being a closed-form function
    of any single one of them.
    """
    hazard = 0.0006
    hazard += 0.004 * (1.0 - sub.engagement)
    hazard += 0.005 * sub.dissatisfaction
    hazard += 0.0015 * sub.price_sensitivity
    hazard += 0.0035 * min(sub.failures, 4)

    # Dormancy is the strongest churn signal in real subscription businesses,
    # and unlike the latent traits above it is *observable*: it shows up in the
    # session log, so the model can actually learn it.  Without this term the
    # hazard depends only on hidden state and no feature set could do better
    # than guess.
    dormant_days = 0 if sub.last_session_on is None else (day - sub.last_session_on).days
    hazard += 0.0009 * min(dormant_days, 45)

    if not sub.auto_renew:
        hazard *= 2.4
    # Early-tenure subscribers churn far more readily than settled ones.
    if (day - sub.signup_date).days < 60:
        hazard *= 1.7
    if scenario and scenario.active_on(day):
        hazard *= scenario.cancellation_multiplier
    return float(min(hazard, 0.5))


def simulate_events(
    n_subscribers: int = 4_000,
    start: str | date | None = None,
    end: str | date | None = None,
    seed: int = settings.RANDOM_SEED,
    scenario: DriftScenario | None = None,
    engine=None,
    reset: bool = True,
) -> SimulationResult:
    """Simulate a subscriber base over a date range and load it into the warehouse.

    Args:
        n_subscribers: How many subscribers sign up across the window.
        start: First simulated day. Defaults to ``settings.SIMULATION_START``.
        end: Last simulated day. Defaults to ``settings.SIMULATION_END``.
        seed: Seed for reproducibility.
        scenario: Optional behavioural shift, for demonstrating drift.
        engine: Optional SQLAlchemy engine; defaults to the configured one.
        reset: Whether to clear existing rows first.

    Returns:
        A :class:`SimulationResult` with per-table row counts.
    """
    first = _to_date(start or settings.SIMULATION_START)
    last = _to_date(end or settings.SIMULATION_END)
    if last <= first:
        raise ValueError(f"end ({last}) must be after start ({first})")

    rng = np.random.default_rng(seed)
    total_days = (last - first).days

    create_schema(engine)
    if reset:
        truncate_all(engine)

    # Signups are spread across the window so that at any cutoff there is a
    # realistic mix of tenures rather than one uniform cohort.
    signup_offsets = np.sort(rng.integers(0, max(total_days - 30, 1), size=n_subscribers))

    population: list[_Subscriber] = []
    subscriber_rows: list[dict] = []
    for index, offset in enumerate(signup_offsets, start=1):
        signup_day = first + timedelta(days=int(offset))
        plan_pool = PLAN_TYPES
        if scenario and scenario.active_on(signup_day) and rng.random() < scenario.new_plan_share:
            plan_pool = (scenario.new_plan_name,) * 3
        sub = _make_subscriber(index, signup_day, rng, plan_pool)
        population.append(sub)
        subscriber_rows.append(
            {
                "subscriber_id": sub.subscriber_id,
                "signup_date": sub.signup_date,
                "acquisition_channel": str(rng.choice(ACQUISITION_CHANNELS)),
            }
        )

    lifecycle: list[dict] = []
    session_rows: list[dict] = []
    payment_rows: list[dict] = []
    ticket_rows: list[dict] = []

    by_signup: dict[date, list[_Subscriber]] = {}
    for sub in population:
        by_signup.setdefault(sub.signup_date, []).append(sub)

    active: list[_Subscriber] = []
    day = first
    while day <= last:
        joining = by_signup.get(day, [])
        for sub in joining:
            lifecycle.append(
                {
                    "subscriber_id": sub.subscriber_id,
                    "event_type": schema.SIGNUP,
                    "plan_type": sub.plan_type,
                    "monthly_fee": sub.monthly_fee,
                    "is_auto_renew_enabled": sub.auto_renew,
                    "occurred_at": _at(day, rng),
                }
            )
            payment_rows.append(
                {
                    "subscriber_id": sub.subscriber_id,
                    "occurred_at": _at(day, rng),
                    "amount": sub.monthly_fee,
                    "status": "succeeded",
                    "discount_applied": False,
                }
            )
        active.extend(joining)

        drifting = bool(scenario and scenario.active_on(day))

        still_active: list[_Subscriber] = []
        for sub in active:
            # --- sessions -------------------------------------------------
            rate = sub.engagement * 1.4
            if drifting:
                rate *= scenario.engagement_multiplier
            todays_sessions = int(rng.poisson(rate))
            for _ in range(todays_sessions):
                session_rows.append(
                    {
                        "subscriber_id": sub.subscriber_id,
                        "occurred_at": _at(day, rng),
                        "duration_minutes": round(float(rng.gamma(2.0, 12.0)), 2),
                    }
                )
            if todays_sessions:
                sub.last_session_on = day

            # --- support tickets ------------------------------------------
            ticket_rate = 0.004 + 0.02 * sub.dissatisfaction
            if drifting:
                ticket_rate *= scenario.ticket_multiplier
            if rng.random() < ticket_rate:
                ticket_rows.append(
                    {
                        "subscriber_id": sub.subscriber_id,
                        "occurred_at": _at(day, rng),
                        "category": str(rng.choice(TICKET_CATEGORIES)),
                    }
                )

            # --- renewal --------------------------------------------------
            if day >= sub.next_renewal:
                failure_rate = 0.02 + 0.10 * sub.price_sensitivity
                if drifting:
                    failure_rate *= scenario.payment_failure_multiplier
                failed = rng.random() < failure_rate
                discounted = rng.random() < (0.10 + 0.35 * sub.price_sensitivity)
                payment_rows.append(
                    {
                        "subscriber_id": sub.subscriber_id,
                        "occurred_at": _at(day, rng),
                        "amount": round(sub.monthly_fee * (0.8 if discounted else 1.0), 2),
                        "status": "failed" if failed else "succeeded",
                        "discount_applied": bool(discounted),
                    }
                )
                if failed:
                    sub.failures += 1
                    # Retry in a few days rather than waiting a full cycle.
                    sub.next_renewal = day + timedelta(days=3)
                else:
                    sub.next_renewal = day + timedelta(days=30)
                    lifecycle.append(
                        {
                            "subscriber_id": sub.subscriber_id,
                            "event_type": schema.RENEWAL,
                            "plan_type": sub.plan_type,
                            "monthly_fee": sub.monthly_fee,
                            "is_auto_renew_enabled": sub.auto_renew,
                            "occurred_at": _at(day, rng),
                        }
                    )

            # --- cancellation ---------------------------------------------
            if rng.random() < _daily_cancellation_hazard(sub, scenario, day):
                sub.active = False
                sub.cancelled_on = day
                lifecycle.append(
                    {
                        "subscriber_id": sub.subscriber_id,
                        "event_type": schema.CANCELLATION,
                        "plan_type": sub.plan_type,
                        "monthly_fee": sub.monthly_fee,
                        "is_auto_renew_enabled": sub.auto_renew,
                        "occurred_at": _at(day, rng),
                    }
                )
            else:
                still_active.append(sub)

        active = still_active
        day += timedelta(days=1)

    counts = {
        "subscribers": insert_rows(schema.subscribers, subscriber_rows, engine),
        "subscription_events": insert_rows(schema.subscription_events, lifecycle, engine),
        "sessions": insert_rows(schema.sessions, session_rows, engine),
        "payments": insert_rows(schema.payments, payment_rows, engine),
        "support_tickets": insert_rows(schema.support_tickets, ticket_rows, engine),
    }

    return SimulationResult(counts=counts, start=first, end=last, n_subscribers=n_subscribers)


def main() -> None:  # pragma: no cover - CLI convenience
    """Populate the warehouse from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Simulate subscriber events into the warehouse.")
    parser.add_argument("--subscribers", type=int, default=4_000)
    parser.add_argument("--start", default=settings.SIMULATION_START)
    parser.add_argument("--end", default=settings.SIMULATION_END)
    parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = simulate_events(
        n_subscribers=args.subscribers, start=args.start, end=args.end, seed=args.seed
    )
    print(result.summary())


if __name__ == "__main__":  # pragma: no cover
    main()
