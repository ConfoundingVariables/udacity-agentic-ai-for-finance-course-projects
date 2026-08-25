from datetime import date, datetime, timezone
from unittest.mock import Mock

import pytest

from starter.src.compliance_officer_agent import (
    REGULATORY_SOURCES,
    ComplianceOfficerAgent,
)
from starter.src.foundation_sar import (
    AccountData,
    CaseData,
    CustomerData,
    ExplainabilityLogger,
    RiskAnalystOutput,
    TransactionData,
)


def make_case() -> CaseData:
    customer = CustomerData(
        customer_id="CUST_0053",
        name="Tracy Lewis",
        date_of_birth=date(1966, 10, 26),
        ssn_last_4="1828",
        address="8016 Nicole Stream",
        customer_since=date(2024, 5, 11),
        risk_rating="Medium",
        occupation="Nurse",
        annual_income=131_925,
    )
    account = AccountData(
        account_id="CUST_0053_ACC_1",
        customer_id=customer.customer_id,
        account_type="Checking",
        opening_date=date(2024, 7, 23),
        current_balance=17_676.35,
        average_monthly_balance=89_530.85,
        status="Active",
    )
    transactions = [
        TransactionData(
            transaction_id=f"TXN_{number}",
            account_id=account.account_id,
            transaction_date=date(2025, 1, number),
            transaction_type="Cash_Deposit",
            amount=9_500,
            description="Cash deposit",
            location="Branch 12",
            method="Cash",
        )
        for number in range(1, 4)
    ]
    return CaseData(
        case_id="CASE_TEST",
        created_at=datetime.now(timezone.utc),
        customer=customer,
        accounts=[account],
        transactions=transactions,
    )


def risk(
    classification: str = "Structuring", indicators: list[str] | None = None
) -> RiskAnalystOutput:
    return RiskAnalystOutput.model_validate(
        {
            "case_id": "CASE_TEST",
            "suspicious_activity_type": classification,
            "confidence_score": 0.87,
            "risk_level": "High",
            "reasoning": "Five-step risk analysis",
            "suspicious_indicators": indicators
            or ["3 cash deposits from $9,000 to under $10,000"],
        }
    )


def test_deterministic_narrative_fulfills_five_element_contract(tmp_path):
    case = make_case()
    logger = ExplainabilityLogger(tmp_path / "compliance.jsonl")
    agent = ComplianceOfficerAgent(None, logger)

    result = agent.generate_compliance_narrative(case, risk())
    checklist = agent.validate_narrative(case, risk(), result)

    assert result.word_count == len(result.sar_narrative.split()) <= 120
    assert result.regulatory_citations == REGULATORY_SOURCES["Structuring"]
    assert result.completeness_check is True
    assert checklist["complete"] is True
    assert all(label in result.sar_narrative for label in ("Who:", "What:", "When:", "Where:", "Why:"))
    assert logger.entries[-1]["success"] is True


def test_sanctions_narrative_requires_confirmed_authoritative_result(tmp_path):
    case = make_case()
    logger = ExplainabilityLogger(tmp_path / "sanctions.jsonl")
    agent = ComplianceOfficerAgent(None, logger)

    with pytest.raises(ValueError, match="confirmed authoritative OFAC"):
        agent.generate_compliance_narrative(
            case, risk("Sanctions", ["name similarity only"])
        )
    assert logger.entries[-1]["success"] is False

    result = agent.generate_compliance_narrative(
        case, risk("Sanctions", ["Confirmed OFAC SLS screening match"])
    )
    assert result.completeness_check is True
    assert result.regulatory_citations == REGULATORY_SOURCES["Sanctions"]


def test_openai_citations_are_grounded_and_api_failure_is_audited(tmp_path):
    case = make_case()
    client = Mock()
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = (
        '{"narrative":"Tracy Lewis (CUST_0053) used CUST_0053_ACC_1 for '
        'three transactions totaling $28,500 from 2025-01-01 through '
        '2025-01-03 at Branch 12; the repeated cash deposits may indicate '
        'structuring and require human review.",'
        '"narrative_reasoning":"ReACT fact selection",'
        '"regulatory_citations":["invented source"],'
        '"completeness_check":true}'
    )
    client.chat.completions.create.return_value = response
    logger = ExplainabilityLogger(tmp_path / "openai.jsonl")

    result = ComplianceOfficerAgent(client, logger).generate_compliance_narrative(
        case, risk()
    )
    assert result.regulatory_citations == REGULATORY_SOURCES["Structuring"]
    assert "invented source" not in result.regulatory_citations

    failing_client = Mock()
    failing_client.chat.completions.create.side_effect = TimeoutError("timed out")
    failure_logger = ExplainabilityLogger(tmp_path / "api_failure.jsonl")
    with pytest.raises(RuntimeError, match="API request failed"):
        ComplianceOfficerAgent(failing_client, failure_logger).generate_compliance_narrative(
            case, risk()
        )
    assert failure_logger.entries[-1]["success"] is False

    fallback_logger = ExplainabilityLogger(tmp_path / "compliance_fallback.jsonl")
    fallback_result = ComplianceOfficerAgent(
        failing_client,
        fallback_logger,
        fallback_on_error=True,
    ).generate_compliance_narrative(case, risk())
    assert fallback_result.completeness_check is True
    assert [entry["success"] for entry in fallback_logger.entries] == [False, True]
