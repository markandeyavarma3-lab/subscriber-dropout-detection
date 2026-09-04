"""Tests for the Prometheus exposition layer and its deploy configuration.

The configuration tests matter as much as the code ones here. An alert rule
that references a metric nobody exposes is silently dead - it never fires, and
nothing tells you, which is the worst possible failure mode for monitoring.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from prometheus_client import generate_latest

from src.config import settings
from src.monitoring import prometheus
from tests.conftest import HIGH_RISK_SUBSCRIBER, LOW_RISK_SUBSCRIBER

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"


def _value(name: str, **labels: str) -> float:
    """Read a sample through the public registry API.

    Poking `._value` works for a Counter but not a Histogram, and it is private
    either way - `get_sample_value` is the documented accessor and behaves
    consistently across metric types.
    """
    sample = prometheus.REGISTRY.get_sample_value(name, labels or None)
    return 0.0 if sample is None else float(sample)


def _populate_labelled_families() -> None:
    """Touch every labelled metric so it appears in the exposition.

    Counters and gauges with labels only materialise once a label combination
    is used, so both coverage guards below need the same warm-up. Sharing it
    keeps them from drifting apart and disagreeing about what "exposed" means.
    """
    prometheus.refresh_model_source("local")
    prometheus.DRIFT_PSI.labels(feature="tenure_days").set(0.0)
    prometheus.MODEL_FAIRNESS_RATIO.labels(metric="recall_ratio").set(1.0)
    prometheus.record_prediction(0.5, 1, "medium")
    prometheus.record_shadow_comparison(
        type("C", (), {"agreed": True, "divergence": 0.0})()
    )


def _exposed_names() -> set[str]:
    """Every metric family name currently in the registry."""
    return {
        line.split("{")[0].split(" ")[0]
        for line in generate_latest(prometheus.REGISTRY).decode().splitlines()
        if line and not line.startswith("#")
    }


# --------------------------------------------------------------------------- #
# Exposition
# --------------------------------------------------------------------------- #


def test_endpoint_returns_prometheus_text_format(client: TestClient) -> None:
    """The scrape target must speak the exposition format, not JSON."""
    response = client.get("/metrics/prometheus")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "# HELP subscriber_predictions_total" in response.text
    assert "# TYPE subscriber_predictions_total counter" in response.text


def test_json_metrics_endpoint_is_untouched(client: TestClient) -> None:
    """`/metrics` keeps its documented JSON contract.

    Prometheus conventionally owns `/metrics`, but this path had a published
    schema first. Silently switching its content type would break every
    existing consumer to satisfy a default Prometheus lets you configure.
    """
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert "served_total" in response.json()


def test_predictions_increment_the_counter(client: TestClient) -> None:
    """Scoring must move the counter, by risk band."""
    before = _value("subscriber_predictions_total", risk_level="high", predicted_label="1")

    for _ in range(3):
        client.post("/predict", json=HIGH_RISK_SUBSCRIBER)

    after = _value("subscriber_predictions_total", risk_level="high", predicted_label="1")
    assert after == before + 3


def test_batch_predictions_are_counted_individually(client: TestClient) -> None:
    """A batch of five is five predictions, not one."""
    sum_before = _value("subscriber_prediction_probability_sum")
    count_before = _value("subscriber_prediction_probability_count")

    client.post("/predict/batch", json={"subscribers": [LOW_RISK_SUBSCRIBER] * 5})

    assert _value("subscriber_prediction_probability_count") == count_before + 5
    assert _value("subscriber_prediction_probability_sum") >= sum_before


def test_serving_gauges_mirror_the_json_endpoint(client: TestClient) -> None:
    """The two endpoints must never disagree about the same underlying state."""
    client.post("/predict", json=HIGH_RISK_SUBSCRIBER)

    live = client.get("/metrics").json()
    client.get("/metrics/prometheus")

    assert _value("subscriber_flagged_rate") == pytest.approx(live["flagged_rate"])
    assert _value("subscriber_model_loaded") == (1 if live["model_loaded"] else 0)


def test_scrape_succeeds_without_a_model(client_without_model: TestClient) -> None:
    """Monitoring must not fail exactly when the thing it monitors is unhealthy."""
    response = client_without_model.get("/metrics/prometheus")

    assert response.status_code == 200
    assert "subscriber_model_loaded" in response.text


def test_served_from_is_exposed_as_a_one_hot_gauge() -> None:
    """Prometheus has no string values, so the source becomes a labelled 0/1."""
    prometheus.refresh_model_source("registry")

    assert _value("subscriber_model_served_from", source="registry") == 1
    assert _value("subscriber_model_served_from", source="local") == 0


# --------------------------------------------------------------------------- #
# Pipeline metrics, read from the run report
# --------------------------------------------------------------------------- #


def test_pipeline_gauges_are_loaded_from_a_run_report(tmp_path) -> None:
    """A batch job has no endpoint to scrape, so its report becomes the source."""
    report = {
        "finished_at": "2025-06-01T03:00:00+00:00",
        "needs_attention": True,
        "training": {"promoted": False},
        "drift": {
            "available": True,
            "overall_verdict": "significant",
            "n_samples": 412,
            "top_features": [
                {"feature": "avg_session_count_last_30d", "psi": 4.46, "verdict": "significant"},
                {"feature": "tenure_days", "psi": 0.08, "verdict": "stable"},
            ],
        },
    }
    path = tmp_path / "last_pipeline_run.json"
    path.write_text(json.dumps(report))

    assert prometheus.refresh_pipeline_gauges(path) is True
    assert _value("subscriber_drift_verdict") == 2
    assert _value("subscriber_pipeline_needs_attention") == 1
    assert _value("subscriber_pipeline_promoted") == 0
    assert _value("subscriber_drift_sample_size") == 412
    assert _value("subscriber_drift_psi", feature="avg_session_count_last_30d") == 4.46


def test_missing_run_report_is_not_an_error(tmp_path) -> None:
    """Before the first scheduled run there is simply nothing to report."""
    assert prometheus.refresh_pipeline_gauges(tmp_path / "absent.json") is False


def test_corrupt_run_report_is_not_an_error(tmp_path) -> None:
    """A half-written file during a scrape must not take the endpoint down."""
    path = tmp_path / "last_pipeline_run.json"
    path.write_text('{"finished_at": "2025')

    assert prometheus.refresh_pipeline_gauges(path) is False


@pytest.mark.parametrize(
    ("verdict", "code"), [("stable", 0), ("moderate", 1), ("significant", 2)]
)
def test_verdict_encoding_is_ordered(verdict: str, code: int) -> None:
    """The codes must be ordered so `>= 2` expresses 'worse than moderate'.

    A label could match a verdict but could not express a threshold, which is
    what an alert rule needs.
    """
    assert prometheus.VERDICT_CODES[verdict] == code


# --------------------------------------------------------------------------- #
# Deploy configuration
# --------------------------------------------------------------------------- #


def test_prometheus_config_scrapes_the_right_path() -> None:
    """The non-default metrics_path is the whole reason `/metrics` stayed JSON."""
    config = yaml.safe_load((DEPLOY / "prometheus" / "prometheus.yml").read_text())
    api_job = next(j for j in config["scrape_configs"] if j["job_name"] == "subscriber-api")

    assert api_job["metrics_path"] == "/metrics/prometheus"
    assert config["rule_files"] == ["/etc/prometheus/alerts.yml"]


def test_every_alert_references_a_metric_that_exists() -> None:
    """The dead-alert guard.

    An alert on a metric nobody exposes never fires and never complains. This
    catches a renamed metric before it silently disables an alert.
    """
    _populate_labelled_families()
    exposed = _exposed_names()
    rules = yaml.safe_load((DEPLOY / "prometheus" / "alerts.yml").read_text())

    referenced: set[str] = set()
    for group in rules["groups"]:
        for rule in group["rules"]:
            referenced |= set(re.findall(r"subscriber_[a-z_]+", rule["expr"]))

    missing = {
        name
        for name in referenced
        if name not in exposed and not any(e.startswith(name) for e in exposed)
    }
    assert not missing, f"alerts reference metrics that are never exposed: {missing}"


def test_alert_rules_are_well_formed() -> None:
    """Every rule needs a severity and a human-readable summary to be useful."""
    rules = yaml.safe_load((DEPLOY / "prometheus" / "alerts.yml").read_text())

    for group in rules["groups"]:
        for rule in group["rules"]:
            assert rule["labels"]["severity"] in {"critical", "warning", "info"}
            assert rule["annotations"]["summary"]


def test_a_rejected_challenger_is_not_a_page() -> None:
    """The gate rejecting a worse model is the system working correctly.

    Paging on it would teach whoever is on call to ignore the pager.
    """
    rules = yaml.safe_load((DEPLOY / "prometheus" / "alerts.yml").read_text())
    by_name = {r["alert"]: r for g in rules["groups"] for r in g["rules"]}

    assert by_name["PipelineNeedsAttention"]["labels"]["severity"] == "info"
    assert by_name["ModelNotLoaded"]["labels"]["severity"] == "critical"


def test_grafana_dashboard_is_valid_and_provisioned() -> None:
    """The dashboard must parse and be pointed at by the provisioning config."""
    dashboard = json.loads(
        (DEPLOY / "grafana" / "dashboards" / "subscriber-dropout.json").read_text()
    )
    assert dashboard["uid"] == "subscriber-dropout"
    assert len(dashboard["panels"]) > 5

    provisioning = yaml.safe_load(
        (DEPLOY / "grafana" / "provisioning" / "dashboards" / "dashboards.yml").read_text()
    )
    assert provisioning["providers"][0]["options"]["path"] == "/var/lib/grafana/dashboards"


def test_grafana_panels_query_metrics_that_exist() -> None:
    """A panel querying a non-existent metric renders an empty graph forever."""
    _populate_labelled_families()
    exposed = _exposed_names()

    dashboard = json.loads(
        (DEPLOY / "grafana" / "dashboards" / "subscriber-dropout.json").read_text()
    )
    referenced: set[str] = set()
    for panel in dashboard["panels"]:
        for target in panel.get("targets", []):
            referenced |= set(re.findall(r"subscriber_[a-z_]+", target.get("expr", "")))

    missing = {
        name
        for name in referenced
        if name not in exposed and not any(e.startswith(name) for e in exposed)
    }
    assert not missing, f"dashboard panels query metrics that are never exposed: {missing}"


def test_compose_wires_prometheus_to_the_api() -> None:
    """The stack must mount the configs, or Prometheus starts up blind."""
    compose = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "docker-compose.yml").read_text()
    )

    assert {"prometheus", "grafana"} <= set(compose["services"])
    mounts = " ".join(compose["services"]["prometheus"]["volumes"])
    assert "prometheus.yml" in mounts
    assert "alerts.yml" in mounts

    grafana_mounts = " ".join(compose["services"]["grafana"]["volumes"])
    assert "provisioning" in grafana_mounts
    assert "dashboards" in grafana_mounts


def test_prometheus_settings_are_not_hard_coded_in_the_endpoint() -> None:
    """The report location follows settings, so a redirected run still scrapes."""
    assert settings.ARTIFACTS_DIR.name


# --------------------------------------------------------------------------- #
# Decision quality reaches the monitoring stack
# --------------------------------------------------------------------------- #


def test_decision_quality_is_exposed_to_prometheus(tmp_path) -> None:
    """A fairness audit nobody can graph is a file, not monitoring.

    Regression guard: calibration, cost and fairness were computed at training
    time and written only to metrics.json - invisible to every dashboard and
    alert in the project.
    """
    metrics = {
        "decision_quality": {
            "calibration": {"expected_calibration_error": 0.0231},
            "costs": {"savings": 564.0},
            "fairness": {
                "passes": False,
                "selection_rate_ratio": 0.4622,
                "recall_ratio": 0.626,
                "roc_auc_ratio": 0.8995,
            },
        }
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics))

    assert prometheus.refresh_decision_quality(path) is True
    assert _value("subscriber_model_calibration_ece") == pytest.approx(0.0231)
    assert _value("subscriber_model_fairness_passes") == 0
    assert _value("subscriber_model_cost_savings_available") == pytest.approx(564.0)
    assert _value("subscriber_model_fairness_ratio", metric="recall_ratio") == pytest.approx(0.626)


def test_missing_metrics_file_is_not_an_error(tmp_path) -> None:
    """Before the first training run there is simply nothing to report."""
    assert prometheus.refresh_decision_quality(tmp_path / "absent.json") is False


def test_a_failed_decision_quality_report_is_skipped(tmp_path) -> None:
    """Diagnostics that errored must not publish misleading zeros."""
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"decision_quality": {"error": "boom"}}))

    assert prometheus.refresh_decision_quality(path) is False


def test_the_fairness_alert_references_a_real_metric() -> None:
    """The audit is only monitoring if something alerts on it."""
    rules = yaml.safe_load((DEPLOY / "prometheus" / "alerts.yml").read_text())
    by_name = {r["alert"]: r for g in rules["groups"] for r in g["rules"]}

    assert "FairnessDisparity" in by_name
    assert "subscriber_model_fairness_passes" in by_name["FairnessDisparity"]["expr"]


def test_the_stream_scorer_is_scraped_as_its_own_job() -> None:
    """A dead consumer must not be hidden behind a healthy API."""
    config = yaml.safe_load((DEPLOY / "prometheus" / "prometheus.yml").read_text())
    jobs = {j["job_name"] for j in config["scrape_configs"]}

    assert "stream-scorer" in jobs


def test_every_exposed_metric_is_graphed_or_alerted_on() -> None:
    """The reverse of the dead-alert guard.

    That guard catches panels pointing at metrics that do not exist. This
    catches the opposite and equally quiet failure: a metric computed, exported
    and then displayed nowhere - which is how the fairness audit ended up
    invisible to every dashboard in the project.
    """
    _populate_labelled_families()

    families = {
        name.replace("_total", "").replace("_bucket", "").replace("_count", "")
        .replace("_sum", "").replace("_created", "")
        for name in _exposed_names()
        if name.startswith("subscriber_")
    }

    dashboard = json.dumps(
        json.loads((DEPLOY / "grafana" / "dashboards" / "subscriber-dropout.json").read_text())
    )
    alerts = (DEPLOY / "prometheus" / "alerts.yml").read_text()
    surfaced = dashboard + alerts

    unused = {family for family in families if family not in surfaced}
    assert not unused, f"exposed but never graphed or alerted on: {sorted(unused)}"


# --------------------------------------------------------------------------- #
# Alertmanager routing
# --------------------------------------------------------------------------- #


def _alertmanager_config() -> dict:
    return yaml.safe_load((DEPLOY / "alertmanager" / "alertmanager.yml").read_text())


def _alert_rules() -> list[dict]:
    rules = yaml.safe_load((DEPLOY / "prometheus" / "alerts.yml").read_text())
    return [rule for group in rules["groups"] for rule in group["rules"]]


def test_prometheus_hands_firing_alerts_to_alertmanager() -> None:
    """Rules that evaluate but route nowhere are a dashboard, not on-call.

    This is the wire that was missing: for months every rule in alerts.yml was
    correct, referenced a live metric, and notified precisely nobody.
    """
    config = yaml.safe_load((DEPLOY / "prometheus" / "prometheus.yml").read_text())

    targets = [
        target
        for entry in config["alerting"]["alertmanagers"]
        for static in entry["static_configs"]
        for target in static["targets"]
    ]
    assert any("alertmanager" in target for target in targets)


def test_every_alert_severity_reaches_a_defined_receiver() -> None:
    """The dead-route guard, mirroring the dead-alert one.

    An alert labelled with a severity no route matches falls through to the
    catch-all. That is survivable; a route pointing at a receiver that does not
    exist is not - Alertmanager refuses to start, taking the whole alerting
    path down with it.
    """
    config = _alertmanager_config()
    receivers = {receiver["name"] for receiver in config["receivers"]}
    root = config["route"]

    routed = {}
    for child in root["routes"]:
        assert child["receiver"] in receivers, f"route points at unknown {child['receiver']}"
        for matcher in child["matchers"]:
            severity = re.fullmatch(r'severity\s*=\s*"([a-z]+)"', matcher)
            if severity:
                routed[severity.group(1)] = child

    assert root["receiver"] in receivers
    used = {rule["labels"]["severity"] for rule in _alert_rules()}
    assert used <= set(routed), f"severities with no route: {sorted(used - set(routed))}"


def test_critical_alerts_nag_and_informational_ones_do_not() -> None:
    """Repeat intervals encode the whole point of severity levels.

    If `info` repeated as often as `critical`, the severity label would be
    decoration. A rejected challenger repeating hourly is how people learn to
    ignore the channel.
    """
    config = _alertmanager_config()
    by_severity = {
        matcher.split('"')[1]: route
        for route in config["route"]["routes"]
        for matcher in route["matchers"]
        if matcher.startswith("severity")
    }

    def seconds(value: str) -> int:
        unit = {"s": 1, "m": 60, "h": 3600, "d": 86400}[value[-1]]
        return int(value[:-1]) * unit

    critical = seconds(by_severity["critical"]["repeat_interval"])
    warning = seconds(by_severity["warning"]["repeat_interval"])
    info = seconds(by_severity["info"]["repeat_interval"])

    assert critical < warning < info
    # A critical alert should not sit in a grouping window before anyone hears.
    assert seconds(by_severity["critical"]["group_wait"]) == 0


def test_inhibit_rules_name_alerts_that_actually_exist() -> None:
    """A typo in a source matcher silently disables the suppression.

    Nothing errors: the rule simply never matches, and one outage goes back to
    producing five notifications.
    """
    defined = {rule["alert"] for rule in _alert_rules()}

    for inhibit in _alertmanager_config()["inhibit_rules"]:
        for matchers in (inhibit["source_matchers"], inhibit["target_matchers"]):
            for matcher in matchers:
                exact = re.fullmatch(r'alertname\s*=\s*"([A-Za-z]+)"', matcher)
                if exact:
                    assert exact.group(1) in defined, f"unknown alert {exact.group(1)}"
                    continue
                regex = re.fullmatch(r'alertname\s*=~\s*"([A-Za-z|]+)"', matcher)
                if regex:
                    for name in regex.group(1).split("|"):
                        assert name in defined, f"unknown alert {name}"


def test_outage_inhibition_groups_on_a_label_the_targets_carry() -> None:
    """`equal` on a label nothing exports suppresses nothing.

    `component` is attached by the scrape config, not by the rules, so this
    guards the join between two files that are edited independently.
    """
    scrape = yaml.safe_load((DEPLOY / "prometheus" / "prometheus.yml").read_text())
    labelled = {
        label
        for job in scrape["scrape_configs"]
        for static in job.get("static_configs", [])
        for label in static.get("labels", {})
    }

    for inhibit in _alertmanager_config()["inhibit_rules"]:
        for label in inhibit.get("equal", []):
            assert label in labelled, f"inhibit joins on '{label}', which no target sets"


def test_compose_runs_alertmanager_with_its_config_mounted() -> None:
    """An Alertmanager with no config file starts up and routes nothing."""
    compose = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "docker-compose.yml").read_text()
    )

    assert "alertmanager" in compose["services"]
    mounts = " ".join(compose["services"]["alertmanager"]["volumes"])
    assert "alertmanager.yml" in mounts
    assert "alertmanager" in compose["services"]["prometheus"]["depends_on"]


def test_the_router_is_itself_scraped_and_alerted_on() -> None:
    """Nobody watching the watchman is how a stack goes quiet and looks fine."""
    config = yaml.safe_load((DEPLOY / "prometheus" / "prometheus.yml").read_text())
    assert "alertmanager" in {job["job_name"] for job in config["scrape_configs"]}

    by_name = {rule["alert"]: rule for rule in _alert_rules()}
    assert by_name["AlertmanagerDown"]["labels"]["severity"] == "critical"


def test_receivers_ship_without_credentials() -> None:
    """Deliberate: routing is real, delivery is left to whoever deploys it.

    This is a guard against a well-meant commit that pastes a live Slack
    webhook or PagerDuty key into the repository to "finish" the setup.
    """
    raw = (DEPLOY / "alertmanager" / "alertmanager.yml").read_text()
    active = "\n".join(
        line for line in raw.splitlines() if not line.lstrip().startswith("#")
    )

    for secret in ("hooks.slack.com", "routing_key:", "api_url:", "https://"):
        assert secret not in active, f"a credential-shaped value leaked in: {secret}"
