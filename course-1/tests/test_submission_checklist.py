import json
from pathlib import Path

import nbformat

PROJECT = Path(__file__).parents[1]
STARTER = PROJECT / "starter"


def test_required_submission_files_are_complete():
    required = [
        STARTER / "src" / "foundation_sar.py",
        STARTER / "src" / "risk_analyst_agent.py",
        STARTER / "src" / "compliance_officer_agent.py",
        STARTER / "src" / "sar_workflow.py",
        STARTER / "notebooks" / "01_data_exploration.ipynb",
        STARTER / "notebooks" / "02_agent_development.ipynb",
        STARTER / "notebooks" / "03_workflow_integration.ipynb",
        STARTER / "README.md",
        PROJECT / "regulatory_research.md",
        PROJECT / "sar_app" / "server.py",
        PROJECT / "sar_app" / "index.html",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in required)
    assert not (STARTER / "src" / "test_scenarios.py").exists()


def test_submission_notebooks_are_executed_without_placeholders():
    notebooks = [
        STARTER / "notebooks" / "01_data_exploration.ipynb",
        STARTER / "notebooks" / "02_agent_development.ipynb",
        STARTER / "notebooks" / "03_workflow_integration.ipynb",
    ]
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        assert code_cells
        assert all(cell.execution_count is not None for cell in code_cells)
        all_source = "\n".join(cell.source for cell in notebook.cells).lower()
        assert "todo" not in all_source
        assert "to implement" not in all_source

    integration_source = "\n".join(
        cell.source
        for cell in nbformat.read(notebooks[-1], as_version=4).cells
    ).lower()
    assert "human gate 1" in integration_source
    assert "human gate 2" in integration_source
    assert "cost" in integration_source
    assert "metrics" in integration_source


def test_generated_audit_sar_and_metric_artifacts_are_complete():
    audit_files = sorted((STARTER / "outputs" / "audit_logs").glob("*.jsonl"))
    assert audit_files
    entries = [
        json.loads(line)
        for path in audit_files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required_audit_fields = {
        "event_id",
        "timestamp",
        "component",
        "action",
        "case_id",
        "input_data",
        "output_data",
        "processing_time",
        "success",
    }
    assert entries
    assert all(required_audit_fields <= entry.keys() for entry in entries)
    assert any(entry["action"] == "risk_decision" for entry in entries)
    assert any(entry["action"] == "compliance_decision" for entry in entries)

    sar_files = sorted((STARTER / "outputs" / "filed_sars").glob("*.json"))
    assert sar_files
    for path in sar_files:
        sar = json.loads(path.read_text(encoding="utf-8"))
        assert sar["status"] == "approved"
        assert sar["risk_human_decision"]["decision"] == "Approved"
        assert sar["compliance_human_decision"]["decision"] == "Approved"
        compliance = sar["compliance_output"]
        assert compliance["word_count"] == len(compliance["sar_narrative"].split())
        assert compliance["word_count"] <= 120
        assert compliance["regulatory_citations"]

    metrics = json.loads(
        (STARTER / "outputs" / "metrics" / "workflow_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert metrics["cases_analyzed"] >= 1
    assert metrics["compliance_generated"] <= metrics["risk_approved"]
    assert metrics["avoided_compliance_calls"] >= 1
    assert metrics["estimated_cost_saved_usd"] > 0


def test_readme_records_commands_controls_and_evidence():
    readme = (STARTER / "README.md").read_text(encoding="utf-8").lower()
    assert "python -m ruff check ." in readme
    assert "python -m pytest -q" in readme
    assert "scripts/smoke_api.py" in readme
    assert "human risk gate" in readme
    assert "human compliance gate" in readme
    assert "outputs/audit_logs" in readme
    assert "outputs/filed_sars" in readme
    assert "to implement" not in readme
