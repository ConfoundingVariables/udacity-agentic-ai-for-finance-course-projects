"""Exercise the running FastAPI application through both human gates."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_URL = os.getenv("SAR_APP_URL", "http://127.0.0.1:8000").rstrip("/")


def request(path: str, method: str = "GET", payload: dict | None = None):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = Request(
        f"{BASE_URL}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise RuntimeError(f"{method} {path} failed: {exc.code} {detail}") from exc


def main() -> None:
    summary = request("/api/summary")
    if summary["customer_count"] != 150:
        raise AssertionError("unexpected customer count")
    if summary["agent_mode"] != "deterministic":
        raise AssertionError("smoke test requires SAR_AGENT_MODE=deterministic")

    record = request("/api/cases/CUST_0053/analyze", "POST")
    case_id = record["case"]["case_id"]
    if record["stage"] == "awaiting_risk_review":
        record = request(
            f"/api/workflows/{case_id}/risk-decision",
            "POST",
            {
                "decision": "Approved",
                "reviewer": "API Smoke Reviewer",
                "rationale": "Repeated cash-band deposits warrant narrative review",
            },
        )
    if record["stage"] == "awaiting_compliance_review":
        record = request(
            f"/api/workflows/{case_id}/compliance-decision",
            "POST",
            {
                "decision": "Approved",
                "reviewer": "API Smoke Compliance Reviewer",
                "rationale": "Narrative is factual, complete, and source-grounded",
            },
        )

    if record["stage"] != "filed":
        raise AssertionError(f"workflow did not file: {record['stage']}")
    if not record["filed_sar_path"].endswith(f"{case_id}.json"):
        raise AssertionError("filed SAR path does not match case")

    rejected = request("/api/cases/CUST_0001/analyze", "POST")
    rejected_case_id = rejected["case"]["case_id"]
    if rejected["stage"] == "awaiting_risk_review":
        rejected = request(
            f"/api/workflows/{rejected_case_id}/risk-decision",
            "POST",
            {
                "decision": "Rejected",
                "reviewer": "API Smoke Reviewer",
                "rationale": "Observed activity does not support compliance escalation",
            },
        )
    if rejected["stage"] != "risk_rejected":
        raise AssertionError(f"rejection path failed: {rejected['stage']}")
    if rejected["compliance_output"] is not None:
        raise AssertionError("compliance ran for a risk-rejected case")

    metrics = request("/api/metrics")
    audit = request("/api/audit?limit=100")
    outputs = request("/api/outputs")
    if (
        metrics["cases_analyzed"] != 2
        or metrics["compliance_generated"] != 1
        or metrics["filed_sars"] != 1
        or metrics["avoided_compliance_calls"] != 1
        or metrics["estimated_cost_saved_usd"] <= 0
    ):
        raise AssertionError("workflow metrics are inconsistent")
    if not any(entry["action"] == "risk_decision" for entry in audit):
        raise AssertionError("risk human decision was not audited")
    if not any(entry["action"] == "compliance_decision" for entry in audit):
        raise AssertionError("compliance human decision was not audited")
    if not any(output["name"] == f"{case_id}.json" for output in outputs):
        raise AssertionError("filed SAR was not listed")

    print(
        json.dumps(
            {
                "case_id": case_id,
                "stage": record["stage"],
                "classification": record["risk_analysis"][
                    "suspicious_activity_type"
                ],
                "word_count": record["compliance_output"]["word_count"],
                "filed_sar_path": record["filed_sar_path"],
                "audit_events": len(audit),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
