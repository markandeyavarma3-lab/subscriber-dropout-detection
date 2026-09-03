"""Transport abstraction for the streaming scorer, plus an in-memory broker.

The scoring loop is written against the two small protocols below rather than
against a Kafka client. That is the same split used for Prefect in
:mod:`src.orchestration`: the part that can actually be wrong is testable
without the infrastructure, and the part that needs infrastructure is kept thin
enough to read in one sitting.

It buys three things:

*Real tests.* :class:`InMemoryBroker` runs the whole consume-score-produce loop
in-process, including the failure paths - malformed payloads, a dead model,
partial batch failures - none of which are convenient to trigger against a live
broker.

*Substitutability.* Redpanda, Kafka, SQS and Pub/Sub all fit these protocols.
Swapping one for another is a new adapter, not a rewrite of the scorer.

*Honesty about what is verified.* Everything above the protocol is exercised by
the test suite. The Kafka adapter in :mod:`src.streaming.kafka` is not - it
needs a broker this project has never had one of - and keeping it thin is what
makes that acceptable.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Message:
    """One record read from a topic."""

    topic: str
    key: str | None
    value: bytes
    offset: int = 0
    partition: int = 0

    def json(self) -> Any:
        """Decode the payload as JSON.

        Raises:
            ValueError: If the payload is not valid JSON. Callers are expected
                to catch this and dead-letter the message rather than die: one
                bad record must not stop a consumer.
        """
        try:
            return json.loads(self.value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"message at offset {self.offset} is not valid JSON: {exc}") from exc


class Consumer(Protocol):
    """The read side of a topic."""

    def poll(self, max_records: int, timeout: float) -> list[Message]:
        """Return up to ``max_records`` messages, or an empty list on timeout."""
        ...

    def commit(self) -> None:
        """Mark everything returned so far as processed."""
        ...

    def close(self) -> None:
        """Release the connection."""
        ...


class Producer(Protocol):
    """The write side of a topic."""

    def send(self, topic: str, key: str | None, value: bytes) -> None:
        """Queue a record for delivery."""
        ...

    def flush(self) -> None:
        """Block until every queued record is delivered."""
        ...

    def close(self) -> None:
        """Release the connection."""
        ...


@dataclass
class InMemoryBroker:
    """A tiny broker for tests: topics are lists, offsets are indices.

    Deliberately not a Kafka simulator. It implements exactly the surface the
    scorer uses, so a test can drive the real loop end to end - including
    poisoned messages and produce failures - without a container.
    """

    topics: dict[str, list[Message]] = field(default_factory=lambda: defaultdict(list))
    committed: dict[str, int] = field(default_factory=dict)
    # When set, the next `send` raises. Lets a test assert that a failed
    # produce does not silently drop the batch's offsets.
    fail_next_send: bool = False

    def publish(self, topic: str, payload: Any, key: str | None = None) -> Message:
        """Append a record to a topic, as a client would."""
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        message = Message(topic=topic, key=key, value=body, offset=len(self.topics[topic]))
        self.topics[topic].append(message)
        return message

    def messages(self, topic: str) -> list[Any]:
        """Decoded payloads on a topic, for assertions."""
        return [json.loads(m.value.decode("utf-8")) for m in self.topics[topic]]

    def consumer(self, topic: str) -> InMemoryConsumer:
        """A consumer positioned at the start of ``topic``."""
        return InMemoryConsumer(self, topic)

    def producer(self) -> InMemoryProducer:
        """A producer writing into this broker."""
        return InMemoryProducer(self)


class InMemoryConsumer:
    """Reads a topic in order, tracking an uncommitted position."""

    def __init__(self, broker: InMemoryBroker, topic: str) -> None:
        self._broker = broker
        self._topic = topic
        self._position = 0
        self.closed = False

    def poll(self, max_records: int, timeout: float = 0.0) -> list[Message]:  # noqa: ARG002
        """Return the next slice of the topic."""
        records = self._broker.topics[self._topic][self._position : self._position + max_records]
        self._position += len(records)
        return list(records)

    def commit(self) -> None:
        """Record how far this consumer has acknowledged."""
        self._broker.committed[self._topic] = self._position

    def close(self) -> None:
        self.closed = True


class InMemoryProducer:
    """Writes into the broker, with an opt-in failure for testing."""

    def __init__(self, broker: InMemoryBroker) -> None:
        self._broker = broker
        self._pending: deque[tuple[str, str | None, bytes]] = deque()
        self.closed = False

    def send(self, topic: str, key: str | None, value: bytes) -> None:
        if self._broker.fail_next_send:
            self._broker.fail_next_send = False
            raise RuntimeError("broker unavailable")
        self._pending.append((topic, key, value))

    def flush(self) -> None:
        while self._pending:
            topic, key, value = self._pending.popleft()
            self._broker.publish(topic, value, key=key)

    def close(self) -> None:
        self.flush()
        self.closed = True
