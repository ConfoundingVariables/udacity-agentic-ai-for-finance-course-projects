"""Unit tests for the named batch orchestrator (offline)."""

from agents.orchestrator_worker import OrchestratorWorkerPattern


SAMPLE = [
    {"message_id": "M1", "amount": "60000.00 USD", "message_type": "MT103"},
    {"message_id": "M2", "amount": "900.00 EUR", "message_type": "MT202"},
    {"message_id": "M3", "currency": "USD", "amount": "12.00", "message_type": "MT103"},
]


def test_offline_plan_is_named_currency_partition():
    plan = OrchestratorWorkerPattern.Orchestrator().create_grouping_plan(SAMPLE)
    assert set(plan) == {"analysis", "grouping_dimension", "groups"}
    assert plan["grouping_dimension"] == "currency"
    assert {group["name"] for group in plan["groups"]} == {"USD transactions", "EUR transactions"}
    grouped_ids = [message_id for group in plan["groups"] for message_id in group["message_ids"]]
    assert grouped_ids == ["M1", "M3", "M2"]
    assert len(grouped_ids) == len(set(grouped_ids)) == len(SAMPLE)


def test_duplicate_message_ids_are_rejected():
    try:
        OrchestratorWorkerPattern.Orchestrator().create_grouping_plan([
            {"message_id": "M1", "amount": "1 USD"},
            {"message_id": "M1", "amount": "2 USD"},
        ])
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate message IDs must be rejected")


def test_model_plan_invalid_partition_uses_currency_fallback(monkeypatch):
    raw = {
        "analysis": "Invalid partition",
        "grouping_dimension": "message_type",
        "groups": [{"group_id": "a", "name": "First", "message_ids": ["M1", "M1"]}],
    }
    monkeypatch.setattr(
        "agents.orchestrator_worker.llm_client.chat_json",
        lambda *args, **kwargs: raw,
    )
    plan = OrchestratorWorkerPattern.Orchestrator().create_grouping_plan(SAMPLE)
    assert plan["grouping_dimension"] == "currency"
    grouped_ids = [message_id for group in plan["groups"] for message_id in group["message_ids"]]
    assert grouped_ids == ["M1", "M3", "M2"]


def test_worker_routing_uses_grouping_dimension():
    pattern = OrchestratorWorkerPattern()
    workers = pattern._build_worker_pool()
    assert pattern._assign_worker({"grouping_dimension": "currency"}, workers).name == "finance-worker"
    assert pattern._assign_worker({"grouping_dimension": "risk_profile"}, workers).name == "fraud-worker"
    assert pattern._assign_worker({"grouping_dimension": "unfamiliar_dimension"}, workers).name == "generalist-worker"


def test_group_worker_reports_actual_subset_offline():
    agent = OrchestratorWorkerPattern.GenericAgent("finance-worker", ["currency"])
    subset = [{"message_id": "M1", "amount": "60000.00 USD", "message_type": "MT103"}]
    result = agent.process_group({
        "group_id": "g1",
        "group_name": "USD transactions",
        "grouping_dimension": "currency",
        "message_ids": ["M1"],
        "messages": subset,
    })
    assert result["status"] == "completed"
    assert result["messages"] == subset
    assert result["message_ids"] == ["M1"]
    assert result["results"]["amount_totals_by_currency"] == {"USD": 60000.0}
    assert result["results"]["message_type_distribution"] == {"MT103": 1}
    assert result["results"]["high_value_ids"] == ["M1"]


def test_process_with_orchestrator_returns_group_results():
    result = OrchestratorWorkerPattern().process_with_orchestrator(SAMPLE)
    assert "orchestrator_analysis" in result
    assert "group_results" in result
    assert "task_results" not in result
    assert len(result["group_results"]) == 2
    assert all(group["status"] == "completed" for group in result["group_results"])
    assert {message_id for group in result["group_results"] for message_id in group["message_ids"]} == {"M1", "M2", "M3"}


def test_process_with_orchestrator_empty():
    result = OrchestratorWorkerPattern().process_with_orchestrator([])
    assert result["group_results"] == []
    assert result["orchestrator_analysis"]["groups"] == []
