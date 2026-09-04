"""Kafka/Redpanda adapter for the streaming scorer.

Redpanda speaks the Kafka protocol, so the same client works against either.

.. warning::
   **This module is the one part of the streaming stack that is not covered by
   the test suite.** Every test drives
   :class:`src.streaming.transport.InMemoryBroker` instead, because this
   project has never had a broker to run against. Keeping this adapter as thin
   as possible - it does nothing but translate between ``kafka-python`` and the
   two protocols in :mod:`src.streaming.transport` - is what makes that
   acceptable rather than reckless. All the behaviour that can actually be
   wrong (batching, dead-lettering, commit ordering, shutdown) lives above this
   line and is tested.

``kafka-python`` is imported lazily so the rest of the project installs and
runs without it. Streaming is opt-in::

    pip install kafka-python
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import settings
from src.streaming.transport import Message

logger = logging.getLogger(__name__)


class MissingKafkaClientError(RuntimeError):
    """Raised when the streaming extra is used without its dependency."""


def _require_kafka() -> Any:
    """Import ``kafka-python`` with a message that says what to install."""
    try:
        import kafka  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise MissingKafkaClientError(
            "Streaming needs a Kafka client. Install it with `pip install kafka-python`, "
            "or use src.streaming.transport.InMemoryBroker for local testing."
        ) from exc
    return kafka


class KafkaConsumer:
    """Adapts ``kafka.KafkaConsumer`` to the transport protocol."""

    def __init__(
        self,
        topic: str | None = None,
        brokers: str | None = None,
        group_id: str | None = None,
    ) -> None:
        kafka = _require_kafka()
        self.topic = topic or settings.STREAM_INPUT_TOPIC
        self._consumer = kafka.KafkaConsumer(
            self.topic,
            bootstrap_servers=(brokers or settings.STREAM_BROKERS).split(","),
            group_id=group_id or settings.STREAM_CONSUMER_GROUP,
            # Offsets are committed explicitly, only after a successful produce.
            # Auto-commit would acknowledge messages whose scores never reached
            # the output topic - the one way this design can lose data.
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            value_deserializer=None,
        )

    def poll(self, max_records: int, timeout: float) -> list[Message]:
        """Fetch up to ``max_records``, flattening Kafka's per-partition dict."""
        batches = self._consumer.poll(
            timeout_ms=int(timeout * 1000), max_records=max_records
        )
        messages: list[Message] = []
        for partition, records in batches.items():
            for record in records:
                messages.append(
                    Message(
                        topic=record.topic,
                        key=record.key.decode("utf-8") if record.key else None,
                        value=record.value,
                        offset=record.offset,
                        partition=partition.partition,
                    )
                )
        return messages

    def commit(self) -> None:
        self._consumer.commit()

    def close(self) -> None:
        self._consumer.close()


class KafkaProducer:
    """Adapts ``kafka.KafkaProducer`` to the transport protocol."""

    def __init__(self, brokers: str | None = None) -> None:
        kafka = _require_kafka()
        self._producer = kafka.KafkaProducer(
            bootstrap_servers=(brokers or settings.STREAM_BROKERS).split(","),
            # Wait for all in-sync replicas. The runner commits offsets only
            # after flush(), so a weaker ack would let it commit on a write
            # that had not actually landed.
            acks="all",
            retries=3,
            linger_ms=20,
        )

    def send(self, topic: str, key: str | None, value: bytes) -> None:
        self._producer.send(topic, key=key.encode("utf-8") if key else None, value=value)

    def flush(self) -> None:
        self._producer.flush()

    def close(self) -> None:
        self._producer.close()


def build_scorer(**kwargs: Any):  # noqa: ANN201 - avoids importing the runner at module load
    """Construct a :class:`~src.streaming.runner.StreamingScorer` on Kafka."""
    from src.streaming.runner import StreamingScorer

    return StreamingScorer(consumer=KafkaConsumer(), producer=KafkaProducer(), **kwargs)


def main() -> None:  # pragma: no cover - CLI wiring
    """Run the streaming scorer against Kafka/Redpanda until stopped."""
    import argparse

    parser = argparse.ArgumentParser(description="Score subscriber events from a stream.")
    parser.add_argument("--brokers", default=settings.STREAM_BROKERS)
    parser.add_argument("--input-topic", default=settings.STREAM_INPUT_TOPIC)
    parser.add_argument("--output-topic", default=settings.STREAM_OUTPUT_TOPIC)
    parser.add_argument("--batch-size", type=int, default=settings.STREAM_BATCH_SIZE)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--metrics-port", type=int, default=settings.STREAM_METRICS_PORT,
        help="Port to expose Prometheus metrics on.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    from src.streaming.runner import StreamingScorer

    scorer = StreamingScorer(
        consumer=KafkaConsumer(topic=args.input_topic, brokers=args.brokers),
        producer=KafkaProducer(brokers=args.brokers),
        output_topic=args.output_topic,
        batch_size=args.batch_size,
    )
    scorer.install_signal_handlers()
    # Without this, every streamed prediction is invisible to monitoring: the
    # counters increment inside a process nothing can scrape.
    scorer.serve_metrics(args.metrics_port)
    try:
        scorer.run(max_batches=args.max_batches)
    finally:
        scorer.close()


if __name__ == "__main__":  # pragma: no cover
    main()
