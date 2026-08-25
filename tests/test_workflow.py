import json
from pathlib import Path

import pytest

from starter.src.compliance_officer_agent import ComplianceOfficerAgent
from starter.src.foundation_sar import DataLoader, ExplainabilityLogger
from starter.src.risk_analyst_agent import RiskAnalystAgent
from starter.src.sar_workflow import HumanDecisionInput, SarWorkflow

DATA = Path(__file__).parents[1] / "starter" / "data"


def make_workflow(tmp_path) -> SarWorkflow:
    logger = ExplainabilityLogger(tmp_path / "outputs" / "audit_logs" / "audit.jsonl")
    loader = DataLoader.load(DATA)
    return SarWorkflow(
        loader,
        RiskAnalystAgent(None, logger),
        ComplianceOfficerAgent(None, logger),
        logger,
        tmp_path / "outputs",
    )


def decision(value: str, rationale: str) -> HumanDecisionInput:
    return HumanDecisionInput.model_validate(
        {
            "decision": value,
            "reviewer": "Test Reviewer",
            "rationale": rationale,
        }
    )


def test_complete_approved_workflow_generates_sar_and_audit(tmp_path):
    workflow = make_workflow(tmp_path)

    analyzed = workflow.analyze_customer("CUST_0053")
    assert analyzed.stage == "awaiting_risk_review"
    assert analyzed.risk_analysis.suspicious_activity_type == "Structuring"

    compliance_ready = workflow.review_risk(
        analyzed.case.case_id,
        decision("Approved", "Repeated cash-band deposits require narrative review"),
    )
    assert compliance_ready.stage == "awaiting_compliance_review"
    assert compliance_ready.compliance_output is not None

    filed = workflow.review_compliance(
        analyzed.case.case_id,
        decision("Approved", "Narrative is factual, complete, and source-grounded"),
    )
    assert filed.stage == "filed"
    assert filed.filed_sar_path is not None

    document = json.loads(Path(filed.filed_sar_path).read_text(encoding="utf-8"))
    assert document["case_id"] == analyzed.case.case_id
    assert document["status"] == "approved"
    assert document["risk_human_decision"]["decision"] == "Approved"
    assert document["compliance_human_decision"]["decision"] == "Approved"
    assert document["compliance_output"]["word_count"] <= 120

    entries = workflow.audit_entries()
    human_actions = [
        entry["action"]
        for entry in entries
        if entry["component"] == "HumanReview"
    ]
    assert human_actions == ["risk_decision", "compliance_decision"]
    assert workflow.metrics().filed_sars == 1
    assert workflow.metrics_path.exists()
    assert workflow.audit_summary_path.exists()


def test_risk_rejection_skips_compliance_generation(tmp_path):
    workflow = make_workflow(tmp_path)
    analyzed = workflow.analyze_customer("CUST_0001")

    rejected = workflow.review_risk(
        analyzed.case.case_id,
        decision("Rejected", "Observed activity does not justify escalation"),
    )

    assert rejected.stage == "risk_rejected"
    assert rejected.compliance_output is None
    assert rejected.filed_sar_path is None
    assert workflow.metrics().compliance_generated == 0
    assert workflow.metrics().avoided_compliance_calls == 1
    assert workflow.metrics().estimated_cost_saved_usd > 0
    assert not any(
        entry["component"] == "ComplianceOfficer"
        for entry in workflow.audit_entries()
    )


def test_compliance_rejection_and_invalid_transition_do_not_file(tmp_path):
    workflow = make_workflow(tmp_path)
    analyzed = workflow.analyze_customer("CUST_0053")
    workflow.review_risk(
        analyzed.case.case_id,
        decision("Approved", "Escalate for narrative generation"),
    )

    rejected = workflow.review_compliance(
        analyzed.case.case_id,
        decision("Rejected", "Narrative requires additional supporting facts"),
    )
    assert rejected.stage == "compliance_rejected"
    assert rejected.filed_sar_path is None
    assert workflow.list_outputs() == []

    with pytest.raises(ValueError, match="not allowed"):
        workflow.review_compliance(
            analyzed.case.case_id,
            decision("Approved", "Duplicate decision must fail"),
        )
