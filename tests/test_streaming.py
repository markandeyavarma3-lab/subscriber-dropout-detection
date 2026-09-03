"""Tests for the streaming scorer.

A stream is not a request/response API, and the difference is the whole point
of these tests. A malformed HTTP body gets a 422 and the caller's problem stays
the caller's. A malformed *message* sits in the topic forever - a consumer that
dies on it dies on it again on every restart, which is how one bad record takes
a pipeline down for a day.

So most of what follows is about bad input and interrupted runs, not the happy
path. The in-memory broker exists to make exactly those cases testable without
infrastructure.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient  # noqa: F401 - fixtures need the app importable

from src.api import service
from src.streaming import processor
from src.streaming.runner import StreamingScorer
from src.streaming.transport import InMemoryBroker, Message
from tests.conftest import HIGH_RISK_SUBSCRIBER, LOW_RISK_SUBSCRIBER

INPUT = "subscriber-events"
OUTPUT = "subscriber-scores"
DLQ = "subscriber-scores-dlq"


@pytest.fixture()
def broker() -> InMemoryBroker:
    return InMemoryBroker()


@pytest.fixture()
def scorer(broker: InMemoryBroker, trained_model, trained_metadata, monkeypatch):
    """A scorer wired to the in-memory broker, serving the session's model."""
    service.set_model(trained_model.model, trained_model.threshold, trained_metadata)
    runner = StreamingScorer(
        consumer=broker.consumer(INPUT),
        producer=broker.producer(),
        output_topic=OUTPUT,
        dead_letter_topic=DLQ,
        batch_size=10,
    )
    yield runner
    service.reset_model()


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #


def test_events_are_scored_onto_the_output_topic(broker, scorer) -> None:
    """The basic contract: an event in, a risk score out."""
    broker.publish(INPUT, {**HIGH_RISK_SUBSCRIBER, "subscriber_id": "SUB-1"})
    broker.publish(INPUT, {**LOW_RISK_SUBSCRIBER, "subscriber_id": "SUB-2"})

    result = scorer.process_once()

    assert len(result.scored) == 2
    scores = broker.messages(OUTPUT)
    assert {s["subscriber_id"] for s in scores} == {"SUB-1", "SUB-2"}
    for score in scores:
        assert 0.0 <= score["dropout_probability"] <= 1.0
        assert score["risk_level"] in {"low", "medium", "high"}


def test_the_subscriber_id_survives_but_never_reaches_the_model(broker, scorer) -> None:
    """The id is a join key for downstream, never a feature.

    It is carried onto the output so a score can be matched back to a person,
    but `REQUIRED_INPUT_COLUMNS` excludes it, so it cannot be trained or
    predicted on.
    """
    from src.features.build_features import REQUIRED_INPUT_COLUMNS

    broker.publish(INPUT, {**HIGH_RISK_SUBSCRIBER, "subscriber_id": "SUB-9"})
    scorer.process_once()

    assert broker.messages(OUTPUT)[0]["subscriber_id"] == "SUB-9"
    assert "subscriber_id" not in REQUIRED_INPUT_COLUMNS


def test_output_is_keyed_by_subscriber(broker, scorer) -> None:
    """Keying by subscriber keeps one person's scores in order downstream."""
    record = {"subscriber_id": "SUB-7", "dropout_probability": 0.5}
    assert processor.partition_key(record) == "SUB-7"
    assert processor.partition_key({"dropout_probability": 0.5}) is None


def test_a_batch_is_scored_in_one_call(broker, scorer, monkeypatch) -> None:
    """Throughput depends on this: 10 rows must be one predict call, not 10."""
    calls = {"n": 0}
    original = service.predict_batch

    def counting(records, model=None):
        calls["n"] += 1
        return original(records, model=model)

    monkeypatch.setattr(service, "predict_batch", counting)
    monkeypatch.setattr(processor.service, "predict_batch", counting)

    for index in range(10):
        broker.publish(INPUT, {**HIGH_RISK_SUBSCRIBER, "subscriber_id": f"SUB-{index}"})

    scorer.process_once()
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# Bad input must never stop the consumer
# --------------------------------------------------------------------------- #


def test_malformed_json_is_dead_lettered_not_raised(broker, scorer) -> None:
    """The poison-pill case. One bad record must not stop the partition."""
    broker.publish(INPUT, b"{not json at all")
    broker.publish(INPUT, {**HIGH_RISK_SUBSCRIBER, "subscriber_id": "SUB-OK"})

    result = scorer.process_once()

    assert len(result.dead_lettered) == 1
    assert len(result.scored) == 1
    assert broker.messages(OUTPUT)[0]["subscriber_id"] == "SUB-OK"


def test_a_message_missing_fields_is_dead_lettered(broker, scorer) -> None:
    """Incomplete events are named in the dead-letter reason, not just rejected."""
    broker.publish(INPUT, {"subscriber_id": "SUB-X", "tenure_days": 10})

    result = scorer.process_once()

    assert len(result.scored) == 0
    reason = broker.messages(DLQ)[0]["reason"]
    assert "missing required fields" in reason
    assert "plan_type" in reason


def test_a_non_object_payload_is_dead_lettered(broker, scorer) -> None:
    """A JSON array is valid JSON and still unusable as a subscriber."""
    broker.publish(INPUT, [1, 2, 3])

    result = scorer.process_once()

    assert len(result.dead_lettered) == 1
    assert "expected a JSON object" in broker.messages(DLQ)[0]["reason"]


def test_dead_letters_keep_the_original_payload(broker, scorer) -> None:
    """A dead-letter topic you cannot replay from is an expensive log line."""
    broker.publish(INPUT, {"subscriber_id": "SUB-BAD", "tenure_days": 5})

    scorer.process_once()

    record = broker.messages(DLQ)[0]
    assert record["payload"]["subscriber_id"] == "SUB-BAD"
    assert record["offset"] == 0
    assert record["topic"] == INPUT
    assert record["failed_at"]


def test_one_bad_record_does_not_cost_the_whole_batch(broker, scorer) -> None:
    """Valid messages are separated before scoring, not abandoned alongside."""
    broker.publish(INPUT, {**HIGH_RISK_SUBSCRIBER, "subscriber_id": "A"})
    broker.publish(INPUT, b"garbage")
    broker.publish(INPUT, {**LOW_RISK_SUBSCRIBER, "subscriber_id": "B"})

    result = scorer.process_once()

    assert len(result.scored) == 2
    assert len(result.dead_lettered) == 1


# --------------------------------------------------------------------------- #
# Delivery semantics - where streaming pipelines lose data
# --------------------------------------------------------------------------- #


def test_offsets_are_committed_only_after_a_successful_produce(broker, scorer) -> None:
    """At-least-once. Committing first would silently lose scores on a crash."""
    broker.publish(INPUT, {**HIGH_RISK_SUBSCRIBER, "subscriber_id": "SUB-1"})

    scorer.process_once()
    assert broker.committed[INPUT] == 1


def test_a_failed_produce_does_not_commit(broker, scorer) -> None:
    """The one way this design could lose data, guarded explicitly.

    Committing after a failed produce would acknowledge messages whose scores
    never reached the output topic - they would never be re-read, and the
    predictions would be gone with nothing to indicate it.
    """
    broker.publish(INPUT, {**HIGH_RISK_SUBSCRIBER, "subscriber_id": "SUB-1"})
    broker.fail_next_send = True

    scorer.process_once()

    assert broker.committed.get(INPUT, 0) == 0
    assert scorer.stats.produce_failures == 1
    assert broker.messages(OUTPUT) == []


def test_a_missing_model_leaves_messages_uncommitted(broker, monkeypatch) -> None:
    """Good data must not be dead-lettered because a model was briefly absent.

    `reset_model()` alone is not enough to simulate this: it only clears the
    cache, and `get_model()` would happily reload the artifact from disk. The
    model has to be made genuinely unavailable.
    """
    def _unavailable(*_args, **_kwargs):
        raise service.ModelNotLoadedError("no artifact and no registry")

    monkeypatch.setattr(service, "get_model", _unavailable)
    monkeypatch.setattr(service, "load_model", _unavailable)

    runner = StreamingScorer(
        consumer=broker.consumer(INPUT),
        producer=broker.producer(),
        output_topic=OUTPUT,
        dead_letter_topic=DLQ,
    )
    broker.publish(INPUT, {**HIGH_RISK_SUBSCRIBER, "subscriber_id": "SUB-1"})

    result = runner.process_once()

    assert result.retryable is True
    assert result.dead_lettered == []
    assert broker.committed.get(INPUT, 0) == 0


def test_messages_are_rescored_after_the_model_returns(
    broker, trained_model, trained_metadata, monkeypatch
) -> None:
    """The retry actually works: nothing was lost while the model was down."""
    def _unavailable(*_args, **_kwargs):
        raise service.ModelNotLoadedError("no artifact and no registry")

    monkeypatch.setattr(service, "get_model", _unavailable)
    monkeypatch.setattr(service, "load_model", _unavailable)

    runner = StreamingScorer(
        consumer=broker.consumer(INPUT),
        producer=broker.producer(),
        output_topic=OUTPUT,
        dead_letter_topic=DLQ,
    )
    broker.publish(INPUT, {**HIGH_RISK_SUBSCRIBER, "subscriber_id": "SUB-1"})

    assert runner.process_once().retryable is True
    assert broker.committed.get(INPUT, 0) == 0

    # The model comes back; a fresh consumer re-reads the uncommitted message.
    monkeypatch.undo()
    service.set_model(trained_model.model, trained_model.threshold, trained_metadata)
    runner.consumer = broker.consumer(INPUT)
    result = runner.process_once()

    assert len(result.scored) == 1
    assert broker.messages(OUTPUT)[0]["subscriber_id"] == "SUB-1"
    service.reset_model()


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


def test_run_drains_a_topic(broker, scorer) -> None:
    """Several batches in sequence, under a budget."""
    for index in range(25):
        broker.publish(INPUT, {**HIGH_RISK_SUBSCRIBER, "subscriber_id": f"SUB-{index}"})

    stats = scorer.run(max_batches=5)

    assert stats.scored == 25
    assert len(broker.messages(OUTPUT)) == 25


def test_run_stops_when_asked(broker, scorer) -> None:
    """Graceful shutdown finishes the current batch rather than dropping it."""
    for index in range(50):
        broker.publish(INPUT, {**HIGH_RISK_SUBSCRIBER, "subscriber_id": f"SUB-{index}"})

    scorer.request_stop()
    stats = scorer.run(max_batches=10)

    # Stopped before any batch ran, and nothing was half-processed.
    assert stats.polls == 0
    assert broker.messages(OUTPUT) == []


def test_an_empty_topic_is_not_an_error(broker, scorer) -> None:
    """A quiet stream is normal, not a failure."""
    stats = scorer.run(max_batches=3)

    assert stats.messages == 0
    assert stats.scored == 0


def test_close_releases_both_ends(broker, scorer) -> None:
    """Shutdown must flush the producer, not just drop the connection."""
    scorer.close()

    assert scorer.producer.closed is True
    assert scorer.consumer.closed is True


# --------------------------------------------------------------------------- #
# Transport primitives
# --------------------------------------------------------------------------- #


def test_message_json_raises_a_readable_error() -> None:
    """The error names the offset, because that is what you need to find it."""
    message = Message(topic="t", key=None, value=b"\xff\xfe not json", offset=42)

    with pytest.raises(ValueError, match="offset 42"):
        message.json()


def test_encode_round_trips() -> None:
    """Output records must survive serialisation, including odd types."""
    from datetime import datetime

    record = {"subscriber_id": "SUB-1", "scored_at": datetime(2025, 1, 1), "p": 0.5}
    assert json.loads(processor.encode(record))["subscriber_id"] == "SUB-1"


def test_broker_publish_and_read_back(broker) -> None:
    """The test double itself behaves like an ordered log."""
    broker.publish(INPUT, {"a": 1})
    broker.publish(INPUT, {"a": 2})

    consumer = broker.consumer(INPUT)
    first = consumer.poll(1, 0)
    second = consumer.poll(1, 0)

    assert first[0].offset == 0
    assert second[0].offset == 1
    assert consumer.poll(1, 0) == []


# --------------------------------------------------------------------------- #
# Deployment manifests
#
# Structural checks only. There is no cluster here, so these cannot prove the
# manifests deploy - only that they say what they are meant to say, and that
# they stay consistent with the code as it changes.
# --------------------------------------------------------------------------- #


def _k8s_docs(name: str) -> list[dict]:
    import pathlib

    import yaml

    path = pathlib.Path(__file__).resolve().parents[1] / "deploy" / "kubernetes" / name
    return [doc for doc in yaml.safe_load_all(path.read_text()) if doc]


def test_api_deployment_separates_liveness_from_readiness() -> None:
    """The reason /health and /ready are two endpoints.

    A pod with no model is live but not ready: it should leave the load
    balancer, not be restarted into the same state forever.
    """
    deployment = next(d for d in _k8s_docs("api.yaml") if d["kind"] == "Deployment")
    container = deployment["spec"]["template"]["spec"]["containers"][0]

    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["readinessProbe"]["httpGet"]["path"] == "/ready"
    # A slow model load must not read as a hang and get the pod killed.
    assert container["startupProbe"]["failureThreshold"] >= 20


def test_api_deployment_sets_both_requests_and_limits() -> None:
    """Requests without limits starve neighbours; limits without requests blind
    the scheduler."""
    deployment = next(d for d in _k8s_docs("api.yaml") if d["kind"] == "Deployment")
    resources = deployment["spec"]["template"]["spec"]["containers"][0]["resources"]

    assert resources["requests"]["cpu"] and resources["requests"]["memory"]
    assert resources["limits"]["cpu"] and resources["limits"]["memory"]


def test_prometheus_annotation_points_at_the_real_scrape_path() -> None:
    """`/metrics` serves JSON here, so the annotation must not use the default."""
    deployment = next(d for d in _k8s_docs("api.yaml") if d["kind"] == "Deployment")
    annotations = deployment["spec"]["template"]["metadata"]["annotations"]

    assert annotations["prometheus.io/path"] == "/metrics/prometheus"
    assert annotations["prometheus.io/scrape"] == "true"


def test_cluster_config_requires_the_registry() -> None:
    """In a cluster, silently serving a stale baked-in file is worse than
    failing readiness loudly."""
    config = _k8s_docs("config.yaml")[0]["data"]

    assert config["SDD_MODEL_SOURCE"] == "registry"


def test_stream_scorer_has_room_to_finish_its_batch() -> None:
    """The runner commits after producing, so a hard kill loses computed scores."""
    deployment = _k8s_docs("stream-scorer.yaml")[0]

    assert deployment["spec"]["template"]["spec"]["terminationGracePeriodSeconds"] >= 30


def test_only_the_api_is_autoscaled() -> None:
    """A consumer group cannot use more consumers than partitions, so scaling
    the scorer on CPU would just add idle pods."""
    api_kinds = {d["kind"] for d in _k8s_docs("api.yaml")}
    scorer_kinds = {d["kind"] for d in _k8s_docs("stream-scorer.yaml")}

    assert "HorizontalPodAutoscaler" in api_kinds
    assert "HorizontalPodAutoscaler" not in scorer_kinds


def test_compose_wires_the_scorer_to_redpanda() -> None:
    """The streaming service must actually point at the broker."""
    import pathlib

    import yaml

    compose = yaml.safe_load(
        (pathlib.Path(__file__).resolve().parents[1] / "docker-compose.yml").read_text()
    )

    assert {"redpanda", "stream-scorer"} <= set(compose["services"])
    scorer = compose["services"]["stream-scorer"]
    assert scorer["environment"]["SDD_STREAM_BROKERS"] == "redpanda:9092"
    assert scorer["depends_on"]["redpanda"]["condition"] == "service_healthy"


def test_kafka_adapter_fails_with_an_actionable_message(monkeypatch) -> None:
    """Streaming is an optional extra, so the error must say what to install."""
    import builtins

    from src.streaming import kafka

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "kafka":
            raise ImportError("no module named kafka")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    with pytest.raises(kafka.MissingKafkaClientError, match="pip install kafka-python"):
        kafka._require_kafka()
