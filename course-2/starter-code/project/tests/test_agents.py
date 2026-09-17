"""Unit tests for the fraud-detection agents and aggregator (offline)."""

from agents.workflow_agents.base_agents import (
    FraudAmountDetectionAgent,
    FraudPatternDetectionAgent,
    GeographicRiskAgent,
    AIAnomalyDetectionAgent,
    FraudAggAgent,
)


def test_amount_agent_flags_high_and_round_amounts():
    agent = FraudAmountDetectionAgent()
    result = agent.analyze({"amount": "50000.00 USD"})
    assert result["agent"] == "FraudAmountDetectionAgent"
    # 50000 is > 10000 (0.3) and a multiple of 1000 (0.2) -> 0.5
    assert result["risk_score"] >= 0.5
    assert result["fraud_reasons"]


def test_amount_agent_clean_small_amount():
    result = FraudAmountDetectionAgent().analyze({"amount": "123.45 USD"})
    assert result["risk_score"] == 0
    assert result["fraud_reasons"] == []


def test_pattern_agent_flags_test_bic_and_same_bic():
    agent = FraudPatternDetectionAgent()
    result = agent.analyze({
        "sender_bic": "TESTUS33XXX",
        "receiver_bic": "TESTUS33XXX",
        "remittance_info": "urgent secret transfer",
    })
    assert result["risk_score"] == 1.0  # capped
    joined = " ".join(result["fraud_reasons"]).lower()
    assert "same sender and receiver" in joined
    assert "test" in joined


def test_geographic_agent_flags_high_risk_jurisdiction():
    agent = GeographicRiskAgent()
    result = agent.analyze({
        "sender_bic": "CHASUS33XXX",   # US
        "receiver_bic": "BANKRU22XXX",  # RU (high risk)
    })
    assert result["agent"] == "GeographicRiskAgent"
    assert result["risk_score"] > 0
    assert any("RU" in r for r in result["fraud_reasons"])


def test_geographic_agent_same_country_low_risk():
    agent = GeographicRiskAgent()
    result = agent.analyze({
        "sender_bic": "CHASUS33XXX",
        "receiver_bic": "JPMCUS44XXX",
    })
    # Same country, no risky jurisdiction -> no risk
    assert result["risk_score"] == 0


def test_ai_anomaly_agent_offline_heuristic():
    agent = AIAnomalyDetectionAgent()
    # Very large, round amount -> non-zero heuristic even offline.
    big = agent.analyze({"amount": "9000000.00 USD",
                         "sender_bic": "CHASUS33XXX", "receiver_bic": "DEUTDEFFXXX"})
    assert big["risk_score"] > 0
    assert big["ai_escalated"] is False  # offline -> no LLM escalation

    # Clean small amount (leading digit 1, non-round) -> no anomaly.
    small = agent.analyze({"amount": "123.45 USD",
                           "sender_bic": "CHASUS33XXX", "receiver_bic": "DEUTDEFFXXX"})
    assert small["risk_score"] == 0
    assert small["ai_escalated"] is False


def test_aggregator_math_and_threshold():
    agg = FraudAggAgent()
    results = [
        {"agent": "A", "risk_score": 0.8, "fraud_reasons": ["r1"]},
        {"agent": "B", "risk_score": 0.4, "fraud_reasons": ["r2"]},
    ]
    out = agg.aggregate_results(results)
    assert out["total_risk_score"] == 0.6
    assert out["is_fraudulent"] is True
    assert out["confidence"] == 60.0
    assert len(out["aggregated_reasons"]) == 2


def test_aggregator_empty():
    out = FraudAggAgent().aggregate_results([])
    assert out["is_fraudulent"] is False
    assert out["total_risk_score"] == 0
