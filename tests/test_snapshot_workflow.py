"""Staggered workflow contract for provider-backed API snapshots."""
from pathlib import Path

import yaml


WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "live-snapshots.yml"


def test_snapshot_workflow_spreads_provider_jobs_across_weekdays():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    on = workflow[True] if True in workflow else workflow["on"]
    crons = [entry["cron"] for entry in on["schedule"]]

    assert len(crons) >= 6
    assert len(crons) == len(set(crons))
    assert all(cron.endswith("1-5") for cron in crons)

    hours_minutes = [" ".join(cron.split()[:2]) for cron in crons]
    assert len(set(hours_minutes)) == len(hours_minutes)


def test_snapshot_workflow_selects_one_dataset_per_scheduled_run_and_adds_jitter():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    job = workflow["jobs"]["refresh-snapshot"]
    body = "\n".join(step.get("run", "") for step in job["steps"])

    assert "github.event.schedule" in yaml.dump(job)
    assert "/api/snapshots/refresh/$DATASET" in body
    assert "RANDOM" in body and "sleep" in body
    assert "sectors" in body
    assert "options_flow" in body
    assert "curl" in body and "--max-time" in body
    assert '${BACKEND_URL%/}/api/snapshots/refresh/$DATASET' in body
