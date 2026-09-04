"""Publish sample subscriber events onto the input topic.

Without this, bringing the stack up leaves the scorer connected, subscribed and
idle forever - there is no way to demonstrate that the streaming path works,
which makes "verify the deployment" an untestable instruction.

Events are drawn from :func:`src.data.generate.generate_subscribers`, the same
generator the training data comes from, so the payloads carry exactly the
columns ``REQUIRED_INPUT_COLUMNS`` demands rather than a hand-written dict that
drifts the first time a feature is added.

    # a handful, then exit
    python -m src.streaming.produce --count 50

    # keep going, roughly 5 events a second, until Ctrl-C
    python -m src.streaming.produce --count 0 --rate 5

    # deliberately malformed, to watch the dead-letter path work
    python -m src.streaming.produce --count 5 --corrupt
"""

from __future__ import annotations

import argparse
import json
import logging
import time

from src.config import settings

logger = logging.getLogger(__name__)


def sample_events(count: int, seed: int = 0) -> list[dict]:
    """Generate ``count`` scoreable subscriber payloads.

    Reuses the training generator rather than hand-rolling records, so a
    feature added to the model shows up here automatically instead of silently
    producing messages the scorer will dead-letter.
    """
    from src.data.generate import generate_subscribers
    from src.features.build_features import REQUIRED_INPUT_COLUMNS

    frame = generate_subscribers(n_subscribers=max(count, 1), seed=seed)
    records = frame.to_dict("records")

    events = []
    for index, record in enumerate(records[:count]):
        event = {column: record[column] for column in REQUIRED_INPUT_COLUMNS}
        # The scorer keys the output topic by subscriber, so give each event a
        # stable id rather than letting them all land on one partition.
        event["subscriber_id"] = record.get("subscriber_id") or f"SUB-{index:05d}"
        events.append(event)
    return events


def corrupt_events(count: int) -> list[bytes]:
    """Payloads the scorer must dead-letter rather than crash on.

    Three separate failure modes, because they take different paths through
    ``parse_message``: unparseable bytes, valid JSON of the wrong type, and a
    well-formed object missing required fields. A dead-letter topic that only
    ever sees one of these has not really been tested.
    """
    broken = [
        b"{not json at all",
        json.dumps([1, 2, 3]).encode("utf-8"),
        json.dumps({"subscriber_id": "SUB-BAD", "tenure_days": 10}).encode("utf-8"),
    ]
    return [broken[index % len(broken)] for index in range(count)]


def publish(
    brokers: str,
    topic: str,
    count: int,
    rate: float,
    corrupt: bool,
    seed: int,
) -> int:
    """Send events to the topic, returning how many were published.

    ``count=0`` streams until interrupted, which is what you want when watching
    the Grafana panels move.
    """
    from src.streaming.kafka import KafkaProducer

    producer = KafkaProducer(brokers=brokers)
    delay = 1.0 / rate if rate > 0 else 0.0
    published = 0

    try:
        while True:
            batch = 100 if count == 0 else min(count - published, 100)
            payloads: list[tuple[str | None, bytes]]
            if corrupt:
                payloads = [(None, value) for value in corrupt_events(batch)]
            else:
                payloads = [
                    (event["subscriber_id"], json.dumps(event).encode("utf-8"))
                    for event in sample_events(batch, seed=seed + published)
                ]

            for key, value in payloads:
                producer.send(topic, key, value)
                published += 1
                if delay:
                    # Flush per message when pacing, so the consumer sees a
                    # trickle rather than one lump every hundred events - the
                    # point of --rate is to watch the graphs move.
                    producer.flush()
                    time.sleep(delay)

            producer.flush()
            logger.info("Published %s events to %s", f"{published:,}", topic)

            if count and published >= count:
                return published
    except KeyboardInterrupt:  # pragma: no cover - interactive use
        logger.info("Stopped after %s events", f"{published:,}")
        return published
    finally:
        producer.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish subscriber events to the stream.")
    parser.add_argument("--brokers", default=settings.STREAM_BROKERS)
    parser.add_argument("--topic", default=settings.STREAM_INPUT_TOPIC)
    parser.add_argument(
        "--count", type=int, default=50, help="How many to send; 0 streams until Ctrl-C."
    )
    parser.add_argument(
        "--rate", type=float, default=0.0, help="Events per second; 0 sends as fast as possible."
    )
    parser.add_argument(
        "--corrupt",
        action="store_true",
        help="Send malformed events to exercise the dead-letter topic.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI entry point
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    published = publish(
        brokers=args.brokers,
        topic=args.topic,
        count=args.count,
        rate=args.rate,
        corrupt=args.corrupt,
        seed=args.seed,
    )
    print(f"Published {published:,} events to {args.topic} on {args.brokers}")
    if args.corrupt:
        print(f"These should land on {settings.STREAM_DEAD_LETTER_TOPIC}, not be scored.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
