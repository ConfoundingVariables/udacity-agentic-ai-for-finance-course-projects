import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sar_app.server import app
from starter.src.foundation_sar import (
    AccountData,
    AuditLogEntry,
    ComplianceOfficerOutput,
    DataLoader,
    ExplainabilityLogger,
    RiskAnalystOutput,
)

DATA = Path(__file__).parents[1] / "starter" / "data"


def test_real_data_counts_and_join(tmp_path):
    store = DataLoader.load(DATA, tmp_path / "audit.jsonl")
    assert store.summary().model_dump() == {
        "customer_count": 150,
        "account_count": 178,
        "transaction_count": 4268,
    }
    case = store.get_case("CUST_0001")
    same_case = store.get_case("CUST_0001")
    assert case.case_id == same_case.case_id
    assert case.case_id != case.customer.customer_id
    assert case.created_at.tzinfo is not None
    assert all(a.customer_id == case.customer.customer_id for a in case.accounts)
    assert all(
        t.account_id in {a.account_id for a in case.accounts} for t in case.transactions
    )


def test_validation_rejects_invalid_values():
    with pytest.raises(ValueError):
        AccountData.model_validate(
            {
                "account_id": "a",
                "customer_id": "c",
                "account_type": "x",
                "opening_date": "2020-01-01",
                "current_balance": -1,
                "average_monthly_balance": 0,
                "status": "Active",
            }
        )


def test_loader_reports_missing_and_malformed_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="customers.csv"):
        DataLoader.load(tmp_path)

    (tmp_path / "customers.csv").write_text(
        "customer_id,name,date_of_birth,ssn_last_4,address,phone,customer_since,"
        "risk_rating,occupation,annual_income\n"
        "CUST_BAD,Bad Data,1980-01-01,1234,Somewhere,,2020-01-01,"
        "Unknown,Tester,50000\n",
        encoding="utf-8",
    )
    (tmp_path / "accounts.csv").write_text(
        "account_id,customer_id,account_type,opening_date,current_balance,"
        "average_monthly_balance,status\n",
        encoding="utf-8",
    )
    (tmp_path / "transactions.csv").write_text(
        "transaction_id,account_id,transaction_date,transaction_type,amount,"
        "description,counterparty,location,method\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"customers\.csv row 2"):
        DataLoader.load(tmp_path)


def test_outputs_and_audit_jsonl(tmp_path):
    RiskAnalystOutput(
        case_id="c",
        suspicious_activity_type="Structuring",
        confidence_score=0.5,
        risk_level="Medium",
        reasoning="r",
        suspicious_indicators=[],
    )
    ComplianceOfficerOutput(
        case_id="c",
        sar_narrative="text",
        regulatory_citations=[],
        completeness_check=True,
        reasoning="r",
        word_count=1,
    )
    logger = ExplainabilityLogger(tmp_path / "audit.jsonl")
    entry = logger.log(
        component="test",
        action="check",
        case_id="c",
        input_data={"in": 1},
        output_data={"out": 2},
        confidence_scores={"score": 0.5},
        processing_time=0.1,
        regulatory_flags=["flag"],
    )
    assert isinstance(entry, AuditLogEntry)
    assert entry.event_id.startswith("AUDIT-")
    record = json.loads((tmp_path / "audit.jsonl").read_text())
    assert record["component"] == "test" and record["processing_time"] == 0.1
    assert logger.write_summary(tmp_path / "audit_summary.json")["entry_count"] == 1


def test_api_endpoints():
    with TestClient(app) as client:
        assert client.get("/api/summary").status_code == 200
        assert client.get("/api/cases/CUST_0001").status_code == 200
        assert client.get("/api/cases/NOPE").status_code == 404
        assert client.get("/api/cases?limit=0").status_code == 422
