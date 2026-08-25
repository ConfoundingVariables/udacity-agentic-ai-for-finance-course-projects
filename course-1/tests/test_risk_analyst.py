from datetime import date, datetime, timezone
from unittest.mock import Mock

import pytest

from starter.src.foundation_sar import (
    AccountData,
    CaseData,
    CustomerData,
    ExplainabilityLogger,
    TransactionData,
)
from starter.src.risk_analyst_agent import RiskAnalystAgent


def make_case(transactions: list[TransactionData] | None = None) -> CaseData:
    customer = CustomerData(
        customer_id="CUST_TEST",
        name="Test Customer",
        date_of_birth=date(1980, 1, 1),
        ssn_last_4="1234",
        address="123 Test Street",
        customer_since=date(2020, 1, 1),
        risk_rating="Medium",
        occupation="Consultant",
        annual_income=75_000,
    )
    account = AccountData(
        account_id="ACC_TEST",
        customer_id=customer.customer_id,
        account_type="Checking",
        opening_date=date(2020, 1, 1),
        current_balance=20_000,
        average_monthly_balance=15_000,
        status="Active",
    )
    return CaseData(
        case_id="CASE_TEST",
        created_at=datetime.now(timezone.utc),
        customer=customer,
        accounts=[account],
        transactions=transactions or [],
    )


def transaction(
    number: int,
    *,
    transaction_type: str = "ACH",
    amount: float = 1_000,
    description: str = "Test activity",
    method: str = "ACH",
) -> TransactionData:
    return TransactionData(
        transaction_id=f"TXN_{number}",
        account_id="ACC_TEST",
        transaction_date=date(2025, 1, number),
        transaction_type=transaction_type,
        amount=amount,
        description=description,
        method=method,
    )


@pytest.mark.parametrize(
    "classification",
    ["Structuring", "Sanctions", "Fraud", "Money_Laundering", "Other"],
)
def test_openai_path_validates_all_classifications(tmp_path, classification):
    client = Mock()
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = (
        '{"classification":"'
        + classification
        + '","confidence_score":0.8,"risk_level":"High",'
        '"reasoning":"Five-step evidence summary",'
        '"key_indicators":["observed fact"]}'
    )
    client.chat.completions.create.return_value = response
    logger = ExplainabilityLogger(tmp_path / "risk.jsonl")

    result = RiskAnalystAgent(client, logger).analyze_case(
        make_case([transaction(1)])
    )

    assert result.suspicious_activity_type == classification
    assert result.confidence_score == 0.8
    assert logger.entries[-1]["success"] is True


def test_deterministic_structuring_and_verified_sanctions_rules(tmp_path):
    case = make_case(
        [
            transaction(
                number,
                transaction_type="Cash_Deposit",
                amount=9_500,
                method="Cash",
            )
            for number in range(1, 4)
        ]
    )
    agent = RiskAnalystAgent(
        None, ExplainabilityLogger(tmp_path / "deterministic.jsonl")
    )

    assert agent.analyze_case(case).suspicious_activity_type == "Structuring"
    assert (
        agent.analyze_case(
            case,
            [{"source": "name_similarity", "match_status": "confirmed"}],
        ).suspicious_activity_type
        == "Structuring"
    )
    assert (
        agent.analyze_case(
            case,
            [{"source": "OFAC_SLS", "match_status": "confirmed"}],
        ).suspicious_activity_type
        == "Sanctions"
    )


def test_money_laundering_output_explains_primary_indicator(tmp_path):
    case = make_case(
        [
            transaction(
                number,
                transaction_type="Wire_Transfer",
                amount=20_000,
                method="Wire",
            )
            for number in range(1, 4)
        ]
    )
    result = RiskAnalystAgent(
        None, ExplainabilityLogger(tmp_path / "money_laundering.jsonl")
    ).analyze_case(case)

    assert result.suspicious_activity_type == "Money_Laundering"
    assert result.primary_indicator == "Repeated high-value wire activity"
    assert "movement or layering of funds" in result.reasoning
    assert "totaling" not in result.reasoning


def test_api_and_malformed_output_failures_are_audited(tmp_path):
    case = make_case([transaction(1)])

    failing_client = Mock()
    failing_client.chat.completions.create.side_effect = TimeoutError("timed out")
    api_logger = ExplainabilityLogger(tmp_path / "api_error.jsonl")
    with pytest.raises(RuntimeError, match="API request failed"):
        RiskAnalystAgent(failing_client, api_logger).analyze_case(case)
    assert api_logger.entries[-1]["success"] is False
    assert "API request failed" in api_logger.entries[-1]["reasoning"]

    fallback_logger = ExplainabilityLogger(tmp_path / "risk_fallback.jsonl")
    fallback_result = RiskAnalystAgent(
        failing_client,
        fallback_logger,
        fallback_on_error=True,
    ).analyze_case(case)
    assert fallback_result.suspicious_activity_type == "Other"
    assert [entry["success"] for entry in fallback_logger.entries] == [False, True]

    malformed_client = Mock()
    malformed_response = Mock()
    malformed_response.choices = [Mock()]
    malformed_response.choices[0].message.content = "not json"
    malformed_client.chat.completions.create.return_value = malformed_response
    malformed_logger = ExplainabilityLogger(tmp_path / "json_error.jsonl")
    with pytest.raises(ValueError, match="Failed to parse"):
        RiskAnalystAgent(malformed_client, malformed_logger).analyze_case(case)
    assert malformed_logger.entries[-1]["success"] is False
    assert "JSON parsing failed" in malformed_logger.entries[-1]["reasoning"]
