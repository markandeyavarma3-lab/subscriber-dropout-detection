"""The consume → score → produce loop.

Small on purpose. All the judgement lives in
:mod:`src.streaming.processor`; this owns the ordering, and the ordering is
where streaming pipelines lose data.

Delivery semantics
------------------

**At-least-once**, chosen deliberately. Offsets are committed only after the
scored records have been flushed to the output topic. If the process dies
between scoring and committing, those messages are re-read and re-scored on
restart - a duplicate score, which is harmless here because scoring is
deterministic and the output is keyed by subscriber.

Committing first would be at-most-once, and would silently lose predictions on
any crash. For a churn model that means a subscriber quietly never gets scored,
never gets contacted, and nobody finds out. Duplicates are cheap; silence is
not.
"""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field
from types import FrameType
from typing import Any

from src.config import settings
from src.streaming import processor
from src.streaming.transport import Consumer, Producer

logger = logging.getLogger(__name__)


@dataclass
class RunnerStats:
    """Totals across the lifetime of a runner."""

    polls: int = 0
    messages: int = 0
    scored: int = 0
    dead_lettered: int = 0
    retries: int = 0
    commits: int = 0
    produce_failures: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"polls={self.polls} messages={self.messages} scored={self.scored} "
            f"dead_lettered={self.dead_lettered} retries={self.retries} "
            f"commits={self.commits} produce_failures={self.produce_failures}"
        )


class StreamingScorer:
    """Reads subscriber events, scores them, writes risk scores back."""

    def __init__(
        self,
        consumer: Consumer,
        producer: Producer,
        output_topic: str | None = None,
        dead_letter_topic: str | None = None,
        batch_size: int | None = None,
        poll_timeout: float | None = None,
    ) -> None:
        self.consumer = consumer
        self.producer = producer
        self.output_topic = output_topic or settings.STREAM_OUTPUT_TOPIC
        self.dead_letter_topic = dead_letter_topic or settings.STREAM_DEAD_LETTER_TOPIC
        self.batch_size = batch_size or settings.STREAM_BATCH_SIZE
        self.poll_timeout = (
            poll_timeout if poll_timeout is not None else settings.STREAM_POLL_TIMEOUT
        )
        self.stats = RunnerStats()
        self._stop = False

    def request_stop(self, *_args: Any) -> None:
        """Ask the loop to finish the current batch and exit.

        Signal handlers set a flag rather than raising: killing the process
        mid-batch would drop scores that were computed but never produced, and
        the whole point of committing last is to not do that.
        """
        logger.info("Shutdown requested; finishing the current batch")
        self._stop = True

    def install_signal_handlers(self) -> None:  # pragma: no cover - process-level
        """Turn SIGINT/SIGTERM into a graceful stop."""
        for received in (signal.SIGINT, signal.SIGTERM):
            signal.signal(received, self._handle_signal)

    def _handle_signal(self, _signum: int, _frame: FrameType | None) -> None:  # pragma: no cover
        self.request_stop()

    def process_once(self) -> processor.BatchResult:
        """Poll once, score, produce, and commit if it is safe to.

        Returns the batch result so a caller - or a test - can see exactly what
        happened without reading logs.
        """
        messages = self.consumer.poll(self.batch_size, self.poll_timeout)
        self.stats.polls += 1
        if not messages:
            return processor.BatchResult()

        self.stats.messages += len(messages)
        result = processor.process_batch(messages)

        if result.retryable:
            # The model is missing. Do not commit: these messages are fine and
            # must be re-read once it is back.
            self.stats.retries += 1
            return result

        try:
            for record in result.scored:
                self.producer.send(
                    self.output_topic, processor.partition_key(record), processor.encode(record)
                )
            for record in result.dead_lettered:
                self.producer.send(self.dead_letter_topic, None, processor.encode(record))
            self.producer.flush()
        except Exception:  # noqa: BLE001 - a failed produce must not commit
            # Committing here would acknowledge messages whose scores never
            # reached the output topic - the one way this design can lose data.
            self.stats.produce_failures += 1
            logger.exception("Failed to produce results; not committing this batch")
            return result

        self.consumer.commit()
        self.stats.commits += 1
        self.stats.scored += len(result.scored)
        self.stats.dead_lettered += len(result.dead_lettered)
        logger.info("Batch complete: %s", result.summary())
        return result

    def run(self, max_batches: int | None = None, idle_sleep: float = 0.5) -> RunnerStats:
        """Loop until stopped, or until ``max_batches`` have been processed.

        Args:
            max_batches: Stop after this many polls. ``None`` runs forever,
                which is what a deployed consumer does; tests pass a number.
            idle_sleep: Pause when a poll returns nothing, so an empty topic
                does not spin the CPU.
        """
        batches = 0
        while not self._stop:
            if max_batches is not None and batches >= max_batches:
                break

            result = self.process_once()
            batches += 1

            if result.total == 0 and not result.retryable:
                if max_batches is not None:
                    # Under a batch budget an empty poll means the test topic is
                    # drained; sleeping would just burn the budget on nothing.
                    continue
                time.sleep(idle_sleep)
            elif result.retryable:
                time.sleep(idle_sleep)

        logger.info("Streaming scorer stopped: %s", self.stats.summary())
        return self.stats

    def serve_metrics(self, port: int | None = None) -> bool:
        """Expose this process's Prometheus metrics for scraping.

        The scorer already records every prediction through
        :func:`src.api.service.predict_batch`, which increments the same
        counters the API uses. Without an HTTP server those numbers accumulate
        in a process nothing can reach - every streamed prediction invisible to
        the dashboards, and no way to tell a busy consumer from a dead one.

        Returns ``False`` rather than raising if the port is taken: a metrics
        endpoint failing to bind must not stop the scorer from scoring.
        """
        from prometheus_client import start_http_server

        from src.monitoring.prometheus import REGISTRY

        target = port or settings.STREAM_METRICS_PORT
        try:
            start_http_server(target, registry=REGISTRY)
        except OSError as exc:  # pragma: no cover - depends on the environment
            logger.warning("Could not serve metrics on port %s: %s", target, exc)
            return False

        logger.info("Serving Prometheus metrics on :%s/metrics", target)
        return True

    def close(self) -> None:
        """Flush and release both ends."""
        try:
            self.producer.close()
        finally:
            self.consumer.close()
