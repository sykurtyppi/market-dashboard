"""Shape of .github/workflows/refresh.yml.

Two things this workflow got wrong before are pinned here: a step inserted
with duplicate mapping keys (lenient YAML silently kept the last one, and
GitHub rejected the file), and a green "success" for runs that skipped every
step because the backend was never deployed. The parse is strict, the job
must be gated on a repository variable so an unconfigured schedule shows as
skipped, and an enabled-but-unconfigured run must fail loudly.
"""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "refresh.yml"

EXPECTED_STEPS = [
    "Require configuration",
    "Trigger refresh",
    "Wait for the background refresh to finish",
    "Report failed phases",
    "Gate on data freshness",
    "Check system health",
]


class StrictLoader(yaml.SafeLoader):
    """PyYAML keeps the last duplicate key; GitHub rejects the file. Match GitHub."""


def _no_duplicates(loader, node, deep=False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ValueError(f"duplicate key {key!r} at line {key_node.start_mark.line + 1}")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)


@pytest.fixture(scope="module")
def workflow():
    return yaml.load(WORKFLOW.read_text(), Loader=StrictLoader)


@pytest.fixture(scope="module")
def job(workflow):
    return workflow["jobs"]["refresh"]


def test_parses_strictly_with_no_duplicate_keys(workflow):
    assert workflow["name"] == "Scheduled data refresh"


def test_scheduled_on_weekdays_and_dispatchable(workflow):
    on = workflow[True] if True in workflow else workflow["on"]   # PyYAML reads bare `on` as boolean True
    crons = [entry["cron"] for entry in on["schedule"]]
    assert crons == ["0 14 * * 1-5", "30 21 * * 1-5"]
    assert "workflow_dispatch" in on


def test_job_is_skipped_unless_enabled_or_manual(job):
    # `vars` is allowed in a job-level if; `secrets` is not. An unset variable
    # skips the job, so the run reads "skipped" — never green for doing nothing.
    # Exact, so an operator swap (|| -> &&) — which would make manual runs
    # depend on the variable too — fails here instead of surviving review.
    assert job["if"] == "github.event_name == 'workflow_dispatch' || vars.REFRESH_ENABLED == 'true'"


def test_steps_are_the_expected_ones_in_order(job):
    assert [step["name"] for step in job["steps"]] == EXPECTED_STEPS


def test_enabled_but_unconfigured_fails_loudly(job):
    require = job["steps"][0]["run"]
    assert "exit 1" in require and "::error::" in require
    assert "BACKEND_URL" in require and "MARKET_API_TOKEN" in require


def test_no_step_references_the_retired_skip_output(job):
    for step in job["steps"]:
        assert "steps.cfg" not in yaml.dump(step)
        assert "if" not in step, f"{step['name']} carries a per-step gate; the job-level gate replaced those"


def test_each_step_body_is_valid_bash(job):
    for step in job["steps"]:
        result = subprocess.run(["bash", "-n"], input=step["run"], text=True, capture_output=True)
        assert result.returncode == 0, f"{step['name']}: {result.stderr}"


def test_the_verification_chain_is_intact(job):
    bodies = {step["name"]: step["run"] for step in job["steps"]}
    assert "last_run.error" in bodies["Report failed phases"]
    assert "last_run.failed_phases" in bodies["Report failed phases"]
    assert ".is_fresh" in bodies["Gate on data freshness"]
    assert ".overall_status" in bodies["Check system health"]
