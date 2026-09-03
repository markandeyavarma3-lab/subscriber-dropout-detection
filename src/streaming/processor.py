"""Score a batch of streamed subscriber events.

Transport-free on purpose: this takes messages in and returns records out, so
every failure mode can be tested without a broker. :mod:`src.streaming.runner`
owns the loop; this owns the decisions.

The decisions that matter are all about what to do with bad input, because a
stream is not a request/response API. A malformed HTTP body gets a 422 and the
caller's problem stays the caller's. A malformed *message* sits in the topic
forever, and a consumer that dies on it will die on it again on restart, and
again after that - the classic poison-pill loop that takes a pipeline down for
a day over one bad record.

So nothing here raises on bad data. Every message ends up in exactly one of
three places: scored, dead-lettered with a reason, or - if the model itself is
unavailable - left uncommitted so it can be retried once the model is back.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.api import service
from src.features.build_features import REQUIRED_INPUT_COLUMNS
from src.streaming.transport import Message

logger = logging.getLogger(__name__)

# Carried through from input to output so a downstream consumer can join a
# score back to the subscriber it belongs to. Not a model feature - it is
# deliberately never passed to the pipeline.
ID_FIELD = "subscriber_id"


@dataclass
class BatchResult:
    """What one polled batch produced."""

    scored: list[dict[str, Any]] = field(default_factory=list)
    dead_lettered: list[dict[str, Any]] = field(default_factory=list)
    # Set when the model is unavailable. The caller must NOT commit offsets in
    # that case: the messages are fine, we simply could not score them yet, and
    # dead-lettering good data because a model was briefly missing would be
    # destroying it.
    retryable: bool = False

    @property
    def total(self) -> int:
        return len(self.scored) + len(self.dead_lettered)

    def summary(self) -> str:
        return (
            f"{len(self.scored)} scored, {len(self.dead_lettered)} dead-lettered"
            + (" (retryable)" if self.retryable else "")
        )


def parse_message(message: Message) -> tuple[dict[str, Any] | None, str | None]:
    """Turn a raw message into model input, or explain why it cannot be.

    Returns:
        ``(features, error)`` - exactly one of which is ``None``.
    """
    try:
        payload = message.json()
    except ValueError as exc:
        return None, str(exc)

    if not isinstance(payload, dict):
        return None, f"expected a JSON object, got {type(payload).__name__}"

    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in payload]
    if missing:
        return None, f"missing required fields: {', '.join(missing)}"

    return payload, None


def _dead_letter(message: Message, reason: str) -> dict[str, Any]:
    """Build a dead-letter record.

    Keeps the original payload verbatim. A dead-letter topic whose records
    cannot be replayed once the bug is fixed is just an expensive log line.
    """
    return {
        "reason": reason,
        "topic": message.topic,
        "partition": message.partition,
        "offset": message.offset,
        "key": message.key,
        "failed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Decoded if possible so the record is readable, raw if not.
        "payload": _safe_payload(message),
    }


def _safe_payload(message: Message) -> Any:
    try:
        return message.json()
    except ValueError:
        return message.value.decode("utf-8", errors="replace")


def process_batch(messages: list[Message]) -> BatchResult:
    """Score a batch of messages, separating the unscoreable ones.

    Valid and invalid messages are separated *before* scoring, so one malformed
    record cannot cost the whole batch: the good ones are still scored in a
    single vectorised call rather than being abandoned alongside it.
    """
    result = BatchResult()
    if not messages:
        return result

    scoreable: list[tuple[Message, dict[str, Any]]] = []
    for message in messages:
        features, error = parse_message(message)
        if error is not None:
            logger.warning("Dead-lettering offset %s: %s", message.offset, error)
            result.dead_lettered.append(_dead_letter(message, error))
        else:
            scoreable.append((message, features))

    if not scoreable:
        return result

    records = [
        {column: features[column] for column in REQUIRED_INPUT_COLUMNS}
        for _, features in scoreable
    ]

    try:
        predictions = service.predict_batch(records)
    except service.ModelNotLoadedError:
        # Not the messages' fault. Leave them for a retry rather than
        # dead-lettering perfectly good data because a model was missing.
        logger.warning("No model available; leaving %d messages uncommitted", len(scoreable))
        result.retryable = True
        return result
    except Exception as exc:  # noqa: BLE001 - a bad row must not kill the consumer
        logger.exception("Scoring failed for a batch of %d", len(scoreable))
        for message, _ in scoreable:
            result.dead_lettered.append(_dead_letter(message, f"scoring error: {exc}"))
        return result

    for (message, features), prediction in zip(scoreable, predictions, strict=True):
        result.scored.append(
            {
                ID_FIELD: features.get(ID_FIELD),
                "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source_offset": message.offset,
                **prediction,
            }
        )

    return result


def encode(record: dict[str, Any]) -> bytes:
    """Serialise an output record for the wire."""
    return json.dumps(record, default=str).encode("utf-8")


def partition_key(record: dict[str, Any]) -> str | None:
    """Key output records by subscriber.

    Keying by subscriber puts every score for one person on the same partition,
    which is what keeps their scores in order for a downstream consumer. Without
    it, two scores for the same subscriber can be processed out of order and the
    older one wins.
    """
    value = record.get(ID_FIELD)
    return str(value) if value is not None else None
