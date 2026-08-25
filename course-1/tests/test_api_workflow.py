from fastapi.testclient import TestClient

from sar_app.server import app


def test_interface_matches_design_and_gate_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("SAR_AGENT_MODE", "deterministic")
    monkeypatch.setenv("SAR_OUTPUT_ROOT", str(tmp_path / "outputs"))
    with TestClient(app) as client:
        html = client.get("/").text.lower()

    assert "#f3f0ee" in html
    assert "#141413" in html
    assert "#cf4500" in html
    assert "sofia sans" not in html
    assert "sofiasans" in html
    assert "two agents. two human gates." in html
    assert "risk-decision" in html
    assert "compliance-decision" in html
    assert "approve sar" in html
    assert "innerhtml" not in html
    assert "https://" not in html


def test_complete_api_workflow_with_two_human_gates(monkeypatch, tmp_path):
    monkeypatch.setenv("SAR_AGENT_MODE", "deterministic")
    monkeypatch.setenv("SAR_OUTPUT_ROOT", str(tmp_path / "outputs"))

    with TestClient(app) as client:
        summary = client.get("/api/summary")
        assert summary.status_code == 200
        assert summary.json()["agent_mode"] == "deterministic"

        analyzed = client.post("/api/cases/CUST_0053/analyze")
        assert analyzed.status_code == 200
        record = analyzed.json()
        case_id = record["case"]["case_id"]
        assert record["stage"] == "awaiting_risk_review"
        assert record["risk_analysis"]["suspicious_activity_type"] == "Structuring"

        invalid = client.post(
            f"/api/workflows/{case_id}/risk-decision",
            json={"decision": "Approved", "reviewer": "API Reviewer"},
        )
        assert invalid.status_code == 422

        risk_approved = client.post(
            f"/api/workflows/{case_id}/risk-decision",
            json={
                "decision": "Approved",
                "reviewer": "API Reviewer",
                "rationale": "Repeated cash-band activity requires narrative review",
            },
        )
        assert risk_approved.status_code == 200
        assert risk_approved.json()["stage"] == "awaiting_compliance_review"
        assert risk_approved.json()["compliance_output"]["word_count"] <= 120

        compliance_approved = client.post(
            f"/api/workflows/{case_id}/compliance-decision",
            json={
                "decision": "Approved",
                "reviewer": "Compliance Reviewer",
                "rationale": "Narrative and citations are complete",
            },
        )
        assert compliance_approved.status_code == 200
        filed = compliance_approved.json()
        assert filed["stage"] == "filed"
        assert filed["filed_sar_path"].endswith(f"{case_id}.json")

        metrics = client.get("/api/metrics").json()
        assert metrics["cases_analyzed"] == 1
        assert metrics["compliance_generated"] == 1
        assert metrics["filed_sars"] == 1

        outputs = client.get("/api/outputs").json()
        assert any(output["name"] == f"{case_id}.json" for output in outputs)

        audit = client.get("/api/audit?limit=100").json()
        assert any(entry["action"] == "risk_decision" for entry in audit)
        assert any(entry["action"] == "compliance_decision" for entry in audit)
