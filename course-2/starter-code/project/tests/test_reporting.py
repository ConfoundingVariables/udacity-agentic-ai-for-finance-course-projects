"""Unit tests for reporting and named-group evaluation contracts."""

from evaluator import SolutionEvaluator
from services.reporting import ReportGenerator


def test_compute_metrics_stp_rate(tmp_path):
    generator = ReportGenerator(output_dir=str(tmp_path))
    messages = [
        {"validation_status": "VALID", "fraud_status": "CLEAN", "amount": "100.00 USD"},
        {"validation_status": "VALID", "fraud_status": "FRAUDULENT", "amount": "200.00 USD"},
        {"validation_status": "INVALID", "fraud_status": "CLEAN", "amount": "300.00 USD"},
        {"validation_status": "VALID", "fraud_status": "CLEAN", "amount": "400.00 USD"},
    ]
    metrics = generator.compute_metrics(messages)
    assert metrics["total_messages"] == 4
    assert metrics["valid_messages"] == 3
    assert metrics["fraudulent_messages"] == 1
    assert metrics["stp_count"] == 2
    assert metrics["stp_rate_pct"] == 50.0
    assert metrics["total_value"] == 1000.0


def test_report_retains_named_groups_and_full_evidence(tmp_path):
    generator = ReportGenerator(output_dir=str(tmp_path))
    messages = [{
        "message_id": "A",
        "message_type": "MT103",
        "amount": "100.00 USD",
        "currency": "USD",
        "validation_status": "VALID",
        "fraud_status": "CLEAN",
        "validation_history": [{"status": "VALID"}],
        "correction_evidence": {"changed": False},
        "fraud_analysis": [{"agent": "A", "risk_score": 0}],
    }]
    chain = {"initial_screening": {"A": "clean"}, "final_review": {"batch_summary": {"clean": 1}}}
    orchestrator_sets = [{
        "filter": "all",
        "message_count": 1,
        "result": {
            "orchestrator_analysis": {
                "analysis": "Currency partition",
                "grouping_dimension": "currency",
                "groups": [{"group_id": "g1", "name": "USD transactions", "message_ids": ["A"]}],
            },
            "group_results": [{
                "group_id": "g1",
                "group_name": "USD transactions",
                "message_ids": ["A"],
                "messages": messages,
                "status": "completed",
            }],
            "summary": "one group",
        },
    }]
    report = generator.generate("unit_report", messages, chain, orchestrator_sets)
    assert (tmp_path / "unit_report.json").exists()
    assert (tmp_path / "unit_report.txt").exists()
    assert report["chain_results"] == chain
    grouped = report["orchestrator_sets"][0]
    assert grouped["grouping_dimension"] == "currency"
    assert grouped["groups"][0]["name"] == "USD transactions"
    assert grouped["group_results"][0]["messages"] == messages
    assert report["transactions"][0]["validation_history"]
    assert report["transactions"][0]["correction_evidence"] == {"changed": False}
    assert report["transactions"][0]["fraud_analysis"]


def test_compute_metrics_empty(tmp_path):
    metrics = ReportGenerator(output_dir=str(tmp_path)).compute_metrics([])
    assert metrics["total_messages"] == 0
    assert metrics["stp_rate_pct"] == 0.0


def test_solution_evaluator_full_marks():
    report = {
        "metrics": {"total_messages": 10, "valid_messages": 10, "stp_rate_pct": 90.0},
        "transactions": [{"fraud_status": "CLEAN"} for _ in range(10)],
        "orchestrator_sets": [
            {"filter": "a", "group_count": 1, "group_results": [{"status": "completed"}]},
            {"filter": "b", "group_count": 2, "group_results": [{"status": "completed"}, {"status": "completed"}]},
        ],
    }
    result = SolutionEvaluator().evaluate(report)
    assert result["overall_score"] == 100.0
    assert result["grade"] == "A"


def test_solution_evaluator_penalises_gaps():
    report = {
        "metrics": {"total_messages": 10, "valid_messages": 5, "stp_rate_pct": 50.0},
        "transactions": [{"fraud_status": "PENDING"} for _ in range(10)],
        "orchestrator_sets": [{"filter": "a", "group_count": 1, "group_results": []}],
    }
    result = SolutionEvaluator().evaluate(report)
    assert result["overall_score"] == 40.0
    assert result["grade"] == "F"
