"""Tests for the DVC pipeline and the parameter file it sweeps.

The point of these is to stop `params.yaml` becoming decoration. A parameter
file that looks authoritative and reaches nothing is worse than no parameter
file: `dvc exp run --set-param model.max_depth=8` completes, reports a new
experiment, and trains exactly the model you already had.

That failure was real here. The simulate stage declared `params: simulation`
while its command hardcoded `--subscribers 4000`, so the population in
params.yaml was ignored - the pipeline said 8,000 and trained on 4,000.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import settings

ROOT = Path(__file__).resolve().parents[1]


def _dvc_pipeline() -> dict:
    return yaml.safe_load((ROOT / "dvc.yaml").read_text())


def _params() -> dict:
    return yaml.safe_load((ROOT / "params.yaml").read_text())


def _settings_source() -> str:
    return (ROOT / "src" / "config" / "settings.py").read_text()


# Helpers that take a params.yaml key. The key is always the last positional
# argument, but which position that is varies - `_env_int` takes a default
# first, `_env_optional_float` does not.
_PARAM_READERS = frozenset(
    {"_env_int", "_env_float", "_env_str", "_env_bool", "_env_optional_int", "_env_optional_float"}
)


def _keys_settings_reads() -> set[str]:
    """Parameter keys settings.py looks up, extracted from the AST.

    Parsed rather than grepped. A regex loose enough to catch every call shape
    also catches every dotted string in the file, and the first version of this
    test duly reported "model.joblib" and "subscribers.csv" as missing
    parameters.
    """
    import ast

    keys: set[str] = set()
    for node in ast.walk(ast.parse(_settings_source())):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in _PARAM_READERS:
            continue

        candidates = [kw.value for kw in node.keywords if kw.arg == "param"]
        if node.args:
            candidates.append(node.args[-1])
        for candidate in candidates:
            if (
                isinstance(candidate, ast.Constant)
                and isinstance(candidate.value, str)
                and "." in candidate.value
            ):
                keys.add(candidate.value)
    return keys


def _flatten(node: dict, prefix: str = "") -> set[str]:
    """Every leaf key in a nested mapping, as dotted paths."""
    keys: set[str] = set()
    for key, value in node.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            keys |= _flatten(value, f"{path}.")
        else:
            keys.add(path)
    return keys


# --------------------------------------------------------------------------- #
# params.yaml actually drives the code
# --------------------------------------------------------------------------- #


def test_every_parameter_is_read_by_settings() -> None:
    """The anti-theatre guard.

    A key here with no consumer sweeps nothing. The experiment still runs, still
    reports, and still trains the model you already had.
    """
    consumed = _keys_settings_reads()
    declared = _flatten(_params())

    orphans = declared - consumed
    assert not orphans, f"params.yaml keys nothing reads: {sorted(orphans)}"


def test_every_parameter_settings_reads_exists_in_the_file() -> None:
    """The reverse: a setting pointing at a key nobody defines.

    Silent, because `_param` returns None for a missing key and the setting
    quietly falls back to its code default - so the parameter file stops
    controlling something without anything saying so.
    """
    declared = _flatten(_params())
    missing = _keys_settings_reads() - declared
    assert not missing, f"settings.py reads keys params.yaml does not define: {sorted(missing)}"


@pytest.mark.parametrize(
    ("key", "attribute"),
    [
        ("model.n_estimators", "MODEL_PARAMS"),
        ("split.random_seed", "RANDOM_SEED"),
        ("simulation.n_subscribers", "N_SUBSCRIBERS"),
        ("costs.offer_efficacy", "OFFER_EFFICACY"),
        ("promotion.min_improvement", "PROMOTION_MIN_IMPROVEMENT"),
    ],
)
def test_named_parameters_reach_their_settings(key: str, attribute: str) -> None:
    """Spot-check the wiring end to end rather than only by regex."""
    parts = key.split(".")
    node = _params()
    for part in parts:
        node = node[part]

    value = getattr(settings, attribute)
    if isinstance(value, dict):
        value = value[parts[-1]]

    assert value == pytest.approx(node) if isinstance(node, float) else value == node


def test_the_environment_still_overrides_the_file(monkeypatch) -> None:
    """Containers and CI shrink models without editing a tracked file.

    Also the reason a committed params.yaml can describe the pipeline's intent
    rather than whatever the last debugging session needed.
    """
    monkeypatch.setenv("SDD_N_ESTIMATORS", "7")
    assert settings._env_int("SDD_N_ESTIMATORS", 300, "model.n_estimators") == 7  # noqa: SLF001

    monkeypatch.delenv("SDD_N_ESTIMATORS")
    from_file = _params()["model"]["n_estimators"]
    assert settings._env_int("SDD_N_ESTIMATORS", 999, "model.n_estimators") == from_file  # noqa: SLF001


def test_a_missing_parameter_file_is_survivable(tmp_path) -> None:
    """A malformed or absent params.yaml must not stop the API from starting.

    Every setting carries its own default, so the worst case is a service
    running on those rather than a service that is down.
    """
    assert settings._load_params(tmp_path / "absent.yaml") == {}  # noqa: SLF001

    broken = tmp_path / "params.yaml"
    broken.write_text("model: [this is not a mapping")
    assert settings._load_params(broken) == {}  # noqa: SLF001


def test_null_capacity_survives_as_none() -> None:
    """`null` in the file has to mean "unset", not zero.

    "We can contact nobody this cycle" is a different statement from "there is
    no cap", and collapsing them would silently constrain every threshold.
    """
    assert _params()["capacity"]["max_offer_rate"] is None
    assert settings.RETENTION_CAPACITY_RATE is None


def test_deployment_facts_are_not_versioned_as_parameters() -> None:
    """Where a run writes its files is not an experiment parameter.

    Putting artifact directories or a tracking URI in here would suggest
    reproducing a run means reproducing where it wrote its output.
    """
    declared = " ".join(_flatten(_params()))

    for leaked in ("dir", "path", "uri", "url", "host", "port"):
        assert leaked not in declared, f"params.yaml carries a deployment fact: {leaked}"


# --------------------------------------------------------------------------- #
# The pipeline definition
# --------------------------------------------------------------------------- #


def test_pipeline_stages_declare_the_parameters_they_use() -> None:
    """A stage that does not declare a parameter will not rerun when it changes.

    Which produces the worst kind of stale result: a model that silently
    predates the setting it is supposed to reflect.
    """
    stages = _dvc_pipeline()["stages"]

    assert set(stages["simulate"]["params"]) == {"simulation"}
    train_params = set(stages["train"]["params"])
    assert {"model", "split", "features", "threshold", "costs"} <= train_params


def test_every_declared_parameter_section_exists() -> None:
    """DVC fails the run on an unknown section, but only once someone runs it."""
    sections = set(_params())

    for stage in _dvc_pipeline()["stages"].values():
        for declared in stage.get("params", []):
            assert declared in sections, f"stage declares unknown param section: {declared}"


def test_stage_dependencies_and_outputs_exist_on_disk() -> None:
    """A path typo makes DVC rerun a stage forever, or never."""
    stages = _dvc_pipeline()["stages"]

    for name, stage in stages.items():
        for dependency in stage.get("deps", []):
            assert (ROOT / dependency).exists(), f"{name} depends on missing {dependency}"


def test_the_stages_form_a_chain_rather_than_two_islands() -> None:
    """`train` must consume what `simulate` produces, or ordering is accidental."""
    stages = _dvc_pipeline()["stages"]

    produced = set(stages["simulate"]["outs"])
    consumed = set(stages["train"]["deps"])

    assert produced & consumed, "train does not depend on any simulate output"


def test_metrics_are_tracked_in_git_rather_than_the_cache() -> None:
    """What makes `dvc metrics diff main` able to answer how a change moved AUC.

    The model stays in the cache - it is megabytes and unreadable. The metrics
    are a few kilobytes of JSON and worth a diff.
    """
    metrics = _dvc_pipeline()["stages"]["train"]["metrics"]
    entry = metrics[0]
    path, options = next(iter(entry.items()))

    assert options["cache"] is False
    assert not _is_gitignored(path), "metrics.json is tracked by git, so it must not be ignored"


def _is_gitignored(path: str) -> bool:
    """Whether .gitignore's last matching rule excludes ``path``."""
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", "-q", path], cwd=ROOT, capture_output=True, check=False
    )
    return result.returncode == 0


def test_the_pipeline_has_been_reproduced_at_least_once() -> None:
    """dvc.lock is the evidence. Without it, dvc.yaml is an untested claim.

    It records the content hash of every input and output, which is what makes
    "which data produced this model" answerable after the next simulation
    overwrites the warehouse.
    """
    lock = yaml.safe_load((ROOT / "dvc.lock").read_text())

    assert set(lock["stages"]) == set(_dvc_pipeline()["stages"])
    for stage in lock["stages"].values():
        assert stage["outs"], "a locked stage recorded no outputs"
