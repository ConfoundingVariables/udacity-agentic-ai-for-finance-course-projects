"""Stateful two-agent SAR workflow with explicit human decision gates."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .compliance_officer_agent import ComplianceOfficerAgent
from .foundation_sar import (
    CaseData,
    ComplianceOfficerOutput,
    DataLoader,
    ExplainabilityLogger,
    RiskAnalystOutput,
)
from .risk_analyst_agent import RiskAnalystAgent

DecisionValue = Literal["Approved", "Rejected"]
WorkflowStage = Literal[
    "awaiting_risk_review",
    "risk_rejected",
    "compliance_failed",
    "awaiting_compliance_review",
    "compliance_rejected",
    "filed",
]


class WorkflowModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class HumanDecisionInput(WorkflowModel):
    decision: DecisionValue
    reviewer: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class HumanDecision(HumanDecisionInput):
    decided_at: datetime


class WorkflowRecord(WorkflowModel):
    case: CaseData
    stage: WorkflowStage
    risk_analysis: RiskAnalystOutput
    risk_decision: HumanDecision | None = None
    compliance_output: ComplianceOfficerOutput | None = None
    compliance_decision: HumanDecision | None = None
    filed_sar_path: str | None = None
    risk_processing_ms: float = Field(ge=0)
    compliance_processing_ms: float | None = Field(default=None, ge=0)


class SarDocument(WorkflowModel):
    filing_id: str
    case_id: str
    generated_at: datetime
    status: Literal["approved"]
    subject: dict
    accounts: list[dict]
    transaction_summary: dict
    risk_analysis: RiskAnalystOutput
    risk_human_decision: HumanDecision
    compliance_output: ComplianceOfficerOutput
    compliance_human_decision: HumanDecision
    audit_log: str


class WorkflowMetrics(WorkflowModel):
    cases_analyzed: int = Field(ge=0)
    risk_approved: int = Field(ge=0)
    risk_rejected: int = Field(ge=0)
    compliance_generated: int = Field(ge=0)
    compliance_approved: int = Field(ge=0)
    compliance_rejected: int = Field(ge=0)
    filed_sars: int = Field(ge=0)
    avoided_compliance_calls: int = Field(ge=0)
    average_risk_processing_ms: float = Field(ge=0)
    average_compliance_processing_ms: float = Field(ge=0)
    staged_estimated_cost_usd: float = Field(ge=0)
    unstaged_estimated_cost_usd: float = Field(ge=0)
    estimated_cost_saved_usd: float = Field(ge=0)


class SarWorkflow:
    """Enforce state transitions, human authority, filing, audit, and metrics."""

    def __init__(
        self,
        loader: DataLoader,
        risk_agent: RiskAnalystAgent,
        compliance_agent: ComplianceOfficerAgent,
        logger: ExplainabilityLogger,
        output_root: str | Path,
        *,
        estimated_risk_call_cost_usd: float = 0.002,
        estimated_compliance_call_cost_usd: float = 0.004,
    ) -> None:
        self.loader = loader
        self.risk_agent = risk_agent
        self.compliance_agent = compliance_agent
        self.logger = logger
        self.output_root = Path(output_root)
        self.filed_sars_dir = self.output_root / "filed_sars"
        self.metrics_dir = self.output_root / "metrics"
        self.audit_summary_path = self.output_root / "audit_logs" / "summary.json"
        self.metrics_path = self.metrics_dir / "workflow_metrics.json"
        self.estimated_risk_call_cost_usd = estimated_risk_call_cost_usd
        self.estimated_compliance_call_cost_usd = estimated_compliance_call_cost_usd
        self._records: dict[str, WorkflowRecord] = {}
        self._case_by_customer: dict[str, str] = {}

    def analyze_customer(self, customer_id: str) -> WorkflowRecord:
        existing_case_id = self._case_by_customer.get(customer_id)
        if existing_case_id:
            return self._records[existing_case_id]

        case = self.loader.get_case(customer_id)
        started = perf_counter()
        risk_analysis = self.risk_agent.analyze_case(case)
        elapsed_ms = (perf_counter() - started) * 1000
        record = WorkflowRecord(
            case=case,
            stage="awaiting_risk_review",
            risk_analysis=risk_analysis,
            risk_processing_ms=elapsed_ms,
        )
        self._records[case.case_id] = record
        self._case_by_customer[customer_id] = case.case_id
        self.logger.log(
            component="SarWorkflow",
            action="risk_review_requested",
            case_id=case.case_id,
            input_data={"customer_id": customer_id},
            output_data={
                "stage": record.stage,
                "classification": risk_analysis.suspicious_activity_type,
            },
            processing_time=elapsed_ms / 1000,
            confidence_scores={"risk": risk_analysis.confidence_score},
            regulatory_flags=risk_analysis.suspicious_indicators,
        )
        self._write_derived_outputs()
        return record

    def get_record(self, case_id: str) -> WorkflowRecord:
        try:
            return self._records[case_id]
        except KeyError as exc:
            raise KeyError(f"unknown workflow case: {case_id}") from exc

    def review_risk(
        self, case_id: str, decision_input: HumanDecisionInput
    ) -> WorkflowRecord:
        record = self.get_record(case_id)
        if record.stage != "awaiting_risk_review":
            raise ValueError(
                f"risk decision is not allowed while stage is {record.stage}"
            )

        decision = HumanDecision(
            **decision_input.model_dump(), decided_at=datetime.now(timezone.utc)
        )
        record.risk_decision = decision
        self.logger.log(
            component="HumanReview",
            action="risk_decision",
            case_id=case_id,
            input_data={"stage": "awaiting_risk_review"},
            output_data={"next_stage": "compliance_generation"},
            user_decision=decision.decision,
            regulatory_flags=[decision.rationale],
        )

        if decision.decision == "Rejected":
            record.stage = "risk_rejected"
            self._write_derived_outputs()
            return record

        return self._generate_compliance(record)

    def retry_compliance(self, case_id: str) -> WorkflowRecord:
        record = self.get_record(case_id)
        if record.stage != "compliance_failed":
            raise ValueError(
                f"compliance retry is not allowed while stage is {record.stage}"
            )
        return self._generate_compliance(record)

    def _generate_compliance(self, record: WorkflowRecord) -> WorkflowRecord:
        started = perf_counter()
        try:
            compliance_output = self.compliance_agent.generate_compliance_narrative(
                record.case, record.risk_analysis
            )
        except Exception:
            record.stage = "compliance_failed"
            record.compliance_processing_ms = (perf_counter() - started) * 1000
            self._write_derived_outputs()
            raise

        record.compliance_processing_ms = (perf_counter() - started) * 1000
        record.compliance_output = compliance_output
        record.stage = "awaiting_compliance_review"
        self.logger.log(
            component="SarWorkflow",
            action="compliance_review_requested",
            case_id=record.case.case_id,
            output_data={
                "stage": record.stage,
                "word_count": compliance_output.word_count,
                "completeness_check": compliance_output.completeness_check,
            },
            processing_time=(record.compliance_processing_ms or 0) / 1000,
            regulatory_flags=compliance_output.regulatory_citations,
        )
        self._write_derived_outputs()
        return record

    def review_compliance(
        self, case_id: str, decision_input: HumanDecisionInput
    ) -> WorkflowRecord:
        record = self.get_record(case_id)
        if record.stage != "awaiting_compliance_review":
            raise ValueError(
                f"compliance decision is not allowed while stage is {record.stage}"
            )
        if record.compliance_output is None or record.risk_decision is None:
            raise RuntimeError("workflow state is incomplete")

        decision = HumanDecision(
            **decision_input.model_dump(), decided_at=datetime.now(timezone.utc)
        )
        record.compliance_decision = decision
        self.logger.log(
            component="HumanReview",
            action="compliance_decision",
            case_id=case_id,
            input_data={"stage": "awaiting_compliance_review"},
            output_data={
                "next_stage": (
                    "filed" if decision.decision == "Approved" else "compliance_rejected"
                )
            },
            user_decision=decision.decision,
            regulatory_flags=[decision.rationale],
        )

        if decision.decision == "Rejected":
            record.stage = "compliance_rejected"
            self._write_derived_outputs()
            return record

        record.filed_sar_path = self._write_sar_document(record, decision)
        record.stage = "filed"
        self.logger.log(
            component="SarWorkflow",
            action="sar_document_generated",
            case_id=case_id,
            output_data={"filed_sar_path": record.filed_sar_path},
            user_decision="Approved",
            regulatory_flags=record.compliance_output.regulatory_citations,
        )
        self._write_derived_outputs()
        return record

    def _write_sar_document(
        self, record: WorkflowRecord, compliance_decision: HumanDecision
    ) -> str:
        if record.risk_decision is None or record.compliance_output is None:
            raise RuntimeError("approved decisions and compliance output are required")

        transactions = record.case.transactions
        dates = [transaction.transaction_date for transaction in transactions]
        document = SarDocument(
            filing_id=f"SAR-{uuid4()}",
            case_id=record.case.case_id,
            generated_at=datetime.now(timezone.utc),
            status="approved",
            subject=record.case.customer.model_dump(mode="json"),
            accounts=[
                account.model_dump(mode="json") for account in record.case.accounts
            ],
            transaction_summary={
                "transaction_count": len(transactions),
                "total_amount": round(
                    sum(transaction.amount for transaction in transactions), 2
                ),
                "activity_start": min(dates).isoformat() if dates else None,
                "activity_end": max(dates).isoformat() if dates else None,
            },
            risk_analysis=record.risk_analysis,
            risk_human_decision=record.risk_decision,
            compliance_output=record.compliance_output,
            compliance_human_decision=compliance_decision,
            audit_log=str(self.logger.log_path),
        )
        self.filed_sars_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.filed_sars_dir / f"{record.case.case_id}.json"
        output_path.write_text(
            json.dumps(document.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        return str(output_path)

    def metrics(self) -> WorkflowMetrics:
        records = list(self._records.values())
        risk_approved = sum(
            record.risk_decision is not None
            and record.risk_decision.decision == "Approved"
            for record in records
        )
        risk_rejected = sum(record.stage == "risk_rejected" for record in records)
        compliance_generated = sum(
            record.compliance_output is not None for record in records
        )
        compliance_approved = sum(
            record.compliance_decision is not None
            and record.compliance_decision.decision == "Approved"
            for record in records
        )
        compliance_rejected = sum(
            record.stage == "compliance_rejected" for record in records
        )
        risk_times = [record.risk_processing_ms for record in records]
        compliance_times = [
            record.compliance_processing_ms
            for record in records
            if record.compliance_processing_ms is not None
        ]
        cases_analyzed = len(records)
        avoided_calls = max(0, cases_analyzed - risk_approved)
        staged_cost = (
            cases_analyzed * self.estimated_risk_call_cost_usd
            + compliance_generated * self.estimated_compliance_call_cost_usd
        )
        unstaged_cost = cases_analyzed * (
            self.estimated_risk_call_cost_usd
            + self.estimated_compliance_call_cost_usd
        )
        return WorkflowMetrics(
            cases_analyzed=cases_analyzed,
            risk_approved=risk_approved,
            risk_rejected=risk_rejected,
            compliance_generated=compliance_generated,
            compliance_approved=compliance_approved,
            compliance_rejected=compliance_rejected,
            filed_sars=sum(record.stage == "filed" for record in records),
            avoided_compliance_calls=avoided_calls,
            average_risk_processing_ms=(
                sum(risk_times) / len(risk_times) if risk_times else 0
            ),
            average_compliance_processing_ms=(
                sum(compliance_times) / len(compliance_times)
                if compliance_times
                else 0
            ),
            staged_estimated_cost_usd=round(staged_cost, 6),
            unstaged_estimated_cost_usd=round(unstaged_cost, 6),
            estimated_cost_saved_usd=round(unstaged_cost - staged_cost, 6),
        )

    def audit_entries(self, limit: int = 100) -> list[dict]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if not self.logger.log_path.exists():
            return []
        entries = [
            json.loads(line)
            for line in self.logger.log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return entries[-limit:]

    def list_outputs(self) -> list[dict[str, str]]:
        if not self.filed_sars_dir.exists():
            return []
        outputs: list[dict[str, str]] = []
        for path in sorted(self.filed_sars_dir.glob("*.json")):
            outputs.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "content": path.read_text(encoding="utf-8"),
                }
            )
        return outputs
    def _write_derived_outputs(self) -> None:
        self.logger.write_summary(self.audit_summary_path)
        metrics = self.metrics()
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text(
            json.dumps(metrics.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )


__all__ = [
    "HumanDecision",
    "HumanDecisionInput",
    "SarDocument",
    "SarWorkflow",
    "WorkflowMetrics",
    "WorkflowRecord",
]
