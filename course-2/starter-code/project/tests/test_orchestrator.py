"""Unit tests for the orchestrator-worker pattern (offline)."""

from agents.orchestrator_worker import OrchestratorWorkerPattern


SAMPLE = [
    {'message_id': 'M1', 'amount': '60000.00 USD',
     'sender_bic': 'CHASUS33XXX', 'receiver_bic': 'DEUTDEFFXXX'},
    {'message_id': 'M2', 'amount': '900.00 USD',
     'sender_bic': 'BNPAFRPPXXX', 'receiver_bic': 'BARCGB22XXX'},
]


def test_default_plan_offline():
    orch = OrchestratorWorkerPattern.Orchestrator()
    plan = orch.analyze_and_create_tasks(SAMPLE)
    assert plan['task_count'] == 4
    types = {t['type'] for t in plan['tasks']}
    assert {'compliance_check', 'amount_verification',
            'pattern_detection', 'summary_report'} <= types
    # The high-value (> $50k) message should be captured in the amount task.
    amount_task = next(t for t in plan['tasks'] if t['type'] == 'amount_verification')
    assert 'M1' in amount_task['data']['high_value_ids']
    assert 'M2' not in amount_task['data']['high_value_ids']


def test_worker_capability_routing():
    p = OrchestratorWorkerPattern()
    workers = p._build_worker_pool()
    routes = {
        'compliance_check': 'compliance-worker',
        'fraud_analysis': 'fraud-worker',
        'pattern_detection': 'fraud-worker',
        'amount_verification': 'finance-worker',
        'summary_report': 'reporting-worker',
        'something_unknown': 'generalist-worker',
    }
    for task_type, expected in routes.items():
        worker = p._assign_worker({'type': task_type}, workers)
        assert worker.name == expected, f"{task_type} -> {worker.name}"


def test_generic_agent_execute_offline():
    agent = OrchestratorWorkerPattern.GenericAgent("compliance-worker",
                                                   ["compliance_check"])
    result = agent.execute_task({
        'task_id': 't1', 'type': 'compliance_check',
        'description': 'Screen BICs', 'data': {'x': 1},
    })
    assert result['status'] == 'completed'
    assert result['worker'] == 'compliance-worker'
    assert result['results']['status'] == 'completed'


def test_process_with_orchestrator_end_to_end_offline():
    p = OrchestratorWorkerPattern()
    out = p.process_with_orchestrator(SAMPLE)
    assert 'orchestrator_analysis' in out
    assert out['task_results']
    assert all(r['status'] == 'completed' for r in out['task_results'])


def test_process_with_orchestrator_empty():
    out = OrchestratorWorkerPattern().process_with_orchestrator([])
    assert out['task_results'] == []
