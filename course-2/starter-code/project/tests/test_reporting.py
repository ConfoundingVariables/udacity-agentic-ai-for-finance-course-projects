"""Unit tests for the reporting service and the solution evaluator."""

from services.reporting import ReportGenerator
from evaluator import SolutionEvaluator


def test_compute_metrics_stp_rate(tmp_path):
    gen = ReportGenerator(output_dir=str(tmp_path))
    messages = [
        {'validation_status': 'VALID', 'fraud_status': 'CLEAN', 'amount': '100.00 USD'},
        {'validation_status': 'VALID', 'fraud_status': 'FRAUDULENT', 'amount': '200.00 USD'},
        {'validation_status': 'INVALID', 'fraud_status': 'CLEAN', 'amount': '300.00 USD'},
        {'validation_status': 'VALID', 'fraud_status': 'CLEAN', 'amount': '400.00 USD'},
    ]
    m = gen.compute_metrics(messages)
    assert m['total_messages'] == 4
    assert m['valid_messages'] == 3
    assert m['fraudulent_messages'] == 1
    # STP = VALID and CLEAN -> messages 1 and 4 -> 2/4 = 50%
    assert m['stp_count'] == 2
    assert m['stp_rate_pct'] == 50.0
    assert m['total_value'] == 1000.0


def test_generate_writes_report_files(tmp_path):
    gen = ReportGenerator(output_dir=str(tmp_path))
    messages = [
        {'message_id': 'A', 'message_type': 'MT103', 'amount': '100.00 USD',
         'currency': 'USD', 'validation_status': 'VALID', 'fraud_status': 'CLEAN'},
    ]
    report = gen.generate("unit_report", messages)
    assert (tmp_path / "unit_report.json").exists()
    assert (tmp_path / "unit_report.txt").exists()
    assert report['metrics']['stp_rate_pct'] == 100.0


def test_compute_metrics_empty(tmp_path):
    gen = ReportGenerator(output_dir=str(tmp_path))
    m = gen.compute_metrics([])
    assert m['total_messages'] == 0
    assert m['stp_rate_pct'] == 0.0


def test_solution_evaluator_full_marks():
    report = {
        "metrics": {"total_messages": 10, "valid_messages": 10, "stp_rate_pct": 90.0},
        "transactions": [{"fraud_status": "CLEAN"} for _ in range(10)],
        "orchestrator_sets": [
            {"filter": "a", "task_count": 4},
            {"filter": "b", "task_count": 4},
        ],
    }
    result = SolutionEvaluator().evaluate(report)
    assert result['overall_score'] == 100.0
    assert result['grade'] == 'A'


def test_solution_evaluator_penalises_gaps():
    report = {
        "metrics": {"total_messages": 10, "valid_messages": 5, "stp_rate_pct": 50.0},
        "transactions": [{"fraud_status": "PENDING"} for _ in range(10)],
        "orchestrator_sets": [{"filter": "a", "task_count": 0}],
    }
    result = SolutionEvaluator().evaluate(report)
    # pipeline_completeness(1.0*.2) + validation(0.5*.2) + stp(1.0*.1) = 0.4 -> 40%
    assert result['overall_score'] == 40.0
    assert result['grade'] == 'F'
