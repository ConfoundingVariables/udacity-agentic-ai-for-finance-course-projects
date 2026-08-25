"""ReACT compliance narrative generation grounded in validated case facts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from .foundation_sar import (
    CaseData,
    ComplianceOfficerOutput,
    ExplainabilityLogger,
    RiskAnalystOutput,
    TransactionData,
)

REGULATORY_SOURCES: dict[str, list[str]] = {
    "Structuring": [
        "31 C.F.R. § 1010.311 (currency transaction reports)",
        "31 C.F.R. § 1010.314 (aggregation)",
        "31 C.F.R. § 1020.320(a)(2)(ii) (transactions designed to evade BSA requirements)",
        "FinCEN, Guidance on Preparing a Complete and Sufficient SAR Narrative",
    ],
    "Sanctions": [
        "OFAC Sanctions List Service",
        "31 C.F.R. § 1020.320 (bank suspicious transaction reporting)",
        "FinCEN, Guidance on Preparing a Complete and Sufficient SAR Narrative",
    ],
    "Fraud": [
        "31 C.F.R. § 1020.320(a)(2) (bank suspicious transaction reporting)",
        "FinCEN, Guidance on Preparing a Complete and Sufficient SAR Narrative",
    ],
    "Money_Laundering": [
        "31 U.S.C. § 5311 (BSA purpose)",
        "31 C.F.R. § 1020.320(a)(2) (bank suspicious transaction reporting)",
        "FFIEC BSA/AML Manual, Money Laundering and Terrorist Financing Methods",
    ],
    "Other": [
        "31 C.F.R. § 1020.320(a)(2) (bank suspicious transaction reporting)",
        "FinCEN, Guidance on Preparing a Complete and Sufficient SAR Narrative",
    ],
}


class ComplianceOfficerAgent:
    """Generate one concise, cited narrative from an approved risk analysis."""

    def __init__(
        self,
        openai_client: Any | None,
        explainability_logger: ExplainabilityLogger,
        model: str = "gpt-4",
        *,
        fallback_on_error: bool = False,
    ) -> None:
        self.client = openai_client
        self.logger = explainability_logger
        self.model = model
        self.fallback_on_error = fallback_on_error
        self.system_prompt = """You are a Compliance Officer preparing a BSA/AML SAR narrative for human review. Apply a ReACT framework internally:
REASONING — verify the validated case facts, distinguish observed facts from inference, identify unknowns, and choose only applicable primary-source guidance.
ACTION — select the material who, what, when, where, and why facts; quantify dates, amounts, accounts, channels, counterparties, and locations when available.
CONCLUSION — draft a clear, chronological, factual narrative of no more than 120 words.

Requirements:
- The 120-word maximum is a project constraint, not a federal filing limit.
- Use calibrated wording such as observed, potential, may indicate, and requires human review. Do not declare guilt or criminal intent.
- State unknown or unavailable material facts rather than inventing them.
- Do not claim an OFAC match without authoritative current-list screening evidence.
- For structuring, distinguish the CTR rule from SAR triggers; a below-$10,000 amount alone is not proof.
- Never disclose SAR existence outside the authorized workflow.
- Cite only source names supplied in the prompt.
- Return one JSON object only using: narrative, narrative_reasoning, regulatory_citations, completeness_check.
"""

    def generate_compliance_narrative(
        self,
        case: CaseData,
        risk_analysis: RiskAnalystOutput,
    ) -> ComplianceOfficerOutput:
        """Generate and validate a narrative for one risk-approved case."""

        if not isinstance(case, CaseData):
            raise TypeError("case must be a CaseData instance")
        if not isinstance(risk_analysis, RiskAnalystOutput):
            raise TypeError("risk_analysis must be a RiskAnalystOutput instance")
        if risk_analysis.case_id not in {"UNASSIGNED", case.case_id}:
            raise ValueError("risk analysis does not belong to this case")

        started = perf_counter()
        citations = REGULATORY_SOURCES[risk_analysis.suspicious_activity_type]
        indicator_text = " ".join(risk_analysis.suspicious_indicators).lower()
        if risk_analysis.suspicious_activity_type == "Sanctions" and not (
            "ofac" in indicator_text and "confirm" in indicator_text
        ):
            message = (
                "Sanctions narrative requires confirmed authoritative OFAC "
                "screening evidence"
            )
            self.logger.log_agent_action(
                agent_type="ComplianceOfficer",
                action="generate_narrative",
                case_id=case.case_id,
                input_data={"risk_analysis": risk_analysis.model_dump(mode="json")},
                reasoning=message,
                execution_time_ms=(perf_counter() - started) * 1000,
                success=False,
                error_message=message,
            )
            raise ValueError(message)
        if self.client is None:
            result = self._deterministic_narrative(case, risk_analysis, citations)
            checklist = self.validate_narrative(case, risk_analysis, result)
            self._log_success(
                case,
                risk_analysis,
                result,
                checklist,
                started,
                mode="deterministic",
            )
            return result

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                max_tokens=800,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": self._build_user_prompt(
                            case, risk_analysis, citations
                        ),
                    },
                ],
            )
            content = response.choices[0].message.content
        except Exception as exc:
            elapsed_ms = (perf_counter() - started) * 1000
            self.logger.log_agent_action(
                agent_type="ComplianceOfficer",
                action="generate_narrative",
                case_id=case.case_id,
                input_data={"risk_analysis": risk_analysis.model_dump(mode="json")},
                reasoning=f"API request failed: {exc}",
                execution_time_ms=elapsed_ms,
                success=False,
                error_message=str(exc),
            )
            if self.fallback_on_error:
                result = self._deterministic_narrative(
                    case, risk_analysis, citations
                )
                checklist = self.validate_narrative(case, risk_analysis, result)
                self._log_success(
                    case,
                    risk_analysis,
                    result,
                    checklist,
                    started,
                    mode="deterministic_fallback",
                )
                return result
            raise RuntimeError(f"Compliance Officer API request failed: {exc}") from exc

        try:
            json_text = self._extract_json_from_response(content or "")
            payload = json.loads(json_text)
            narrative = str(payload.get("narrative", payload.get("sar_narrative", "")))
            word_count = len(narrative.split())
            if word_count > 120:
                raise ValueError(
                    f"narrative exceeds 120 word limit ({word_count} words)"
                )
            payload["case_id"] = case.case_id
            payload["word_count"] = word_count
            payload["regulatory_citations"] = list(citations)
            result = ComplianceOfficerOutput.model_validate(payload)
        except (ValueError, TypeError, json.JSONDecodeError, ValidationError) as exc:
            elapsed_ms = (perf_counter() - started) * 1000
            self.logger.log_agent_action(
                agent_type="ComplianceOfficer",
                action="generate_narrative",
                case_id=case.case_id,
                input_data={"risk_analysis": risk_analysis.model_dump(mode="json")},
                reasoning=f"JSON parsing failed: {exc}",
                execution_time_ms=elapsed_ms,
                success=False,
                error_message=str(exc),
            )
            if self.fallback_on_error:
                result = self._deterministic_narrative(
                    case, risk_analysis, citations
                )
                checklist = self.validate_narrative(case, risk_analysis, result)
                self._log_success(
                    case,
                    risk_analysis,
                    result,
                    checklist,
                    started,
                    mode="deterministic_fallback",
                )
                return result
            raise ValueError(
                f"Failed to parse Compliance Officer JSON output: {exc}"
            ) from exc

        checklist = self.validate_narrative(case, risk_analysis, result)
        self._log_success(
            case,
            risk_analysis,
            result,
            checklist,
            started,
            mode="openai",
        )
        return result

    def _deterministic_narrative(
        self,
        case: CaseData,
        risk_analysis: RiskAnalystOutput,
        citations: list[str],
    ) -> ComplianceOfficerOutput:
        transactions = case.transactions
        total = sum(transaction.amount for transaction in transactions)
        dates = [transaction.transaction_date for transaction in transactions]
        start_date = min(dates).isoformat() if dates else "date unavailable"
        end_date = max(dates).isoformat() if dates else "date unavailable"
        account_ids = ", ".join(account.account_id for account in case.accounts[:3])
        if not account_ids:
            account_ids = "account identifiers unavailable"
        locations = sorted(
            {
                transaction.location
                for transaction in transactions
                if transaction.location
            }
        )
        where = ", ".join(locations[:3]) or "transaction location data unavailable"
        indicators = "; ".join(risk_analysis.suspicious_indicators[:2])
        if not indicators:
            indicators = "the automated review identified no specific indicator"

        narrative = (
            f"Who: {case.customer.name} ({case.customer.customer_id}), using "
            f"{account_ids}. What: conducted {len(transactions)} transactions totaling "
            f"${total:,.2f}. When: activity occurred from {start_date} through "
            f"{end_date}. Where: activity involved {where}. Why: observed facts—"
            f"{indicators}—may indicate {risk_analysis.suspicious_activity_type.lower()} "
            "and require human review. Supporting transaction records are retained; "
            "unknown counterparties or locations remain unverified."
        )
        word_count = len(narrative.split())
        if word_count > 120:
            raise ValueError(
                f"narrative exceeds 120 word limit ({word_count} words)"
            )

        reasoning = (
            "ReACT result: verified case and risk facts; selected the material "
            "who/what/when/where/why evidence; used calibrated language; retained "
            "unknowns; and applied only the classification-specific source set."
        )
        candidate = ComplianceOfficerOutput(
            case_id=case.case_id,
            sar_narrative=narrative,
            regulatory_citations=citations,
            completeness_check=False,
            reasoning=reasoning,
            word_count=word_count,
        )
        checklist = self.validate_narrative(case, risk_analysis, candidate)
        return candidate.model_copy(
            update={"completeness_check": checklist["complete"]}
        )

    def validate_narrative(
        self,
        case: CaseData,
        risk_analysis: RiskAnalystOutput,
        result: ComplianceOfficerOutput,
    ) -> dict[str, bool]:
        """Check the observable five-elements, citation, and length contract."""

        narrative = result.sar_narrative
        lowered = narrative.lower()
        account_or_location = any(
            account.account_id in narrative for account in case.accounts
        ) or any(
            transaction.location and transaction.location in narrative
            for transaction in case.transactions
        )
        checks = {
            "who": case.customer.name in narrative
            and case.customer.customer_id in narrative,
            "what": "$" in narrative and "transaction" in lowered,
            "when": any(
                transaction.transaction_date.isoformat() in narrative
                for transaction in case.transactions
            ),
            "where": account_or_location
            or "location data unavailable" in lowered
            or "location identifiers unavailable" in lowered,
            "why": risk_analysis.suspicious_activity_type.lower() in lowered
            or "suspicious" in lowered
            or "may indicate" in lowered,
            "within_word_limit": result.word_count <= 120,
            "has_citations": bool(result.regulatory_citations),
        }
        checks["complete"] = all(checks.values())
        return checks

    def _build_user_prompt(
        self,
        case: CaseData,
        risk_analysis: RiskAnalystOutput,
        citations: Sequence[str],
    ) -> str:
        return (
            "Prepare the final human-review narrative from only these facts.\n"
            f"Case: {case.case_id}\n"
            f"Customer: {case.customer.name} ({case.customer.customer_id})\n"
            f"Accounts: {', '.join(account.account_id for account in case.accounts) or 'not provided'}\n"
            f"Transactions:\n{self._format_transactions_for_compliance(case.transactions)}\n"
            f"Risk analysis: {json.dumps(risk_analysis.model_dump(mode='json'), sort_keys=True)}\n"
            f"Allowed citations: {json.dumps(list(citations))}"
        )

    def _log_success(
        self,
        case: CaseData,
        risk_analysis: RiskAnalystOutput,
        result: ComplianceOfficerOutput,
        checklist: Mapping[str, bool],
        started: float,
        *,
        mode: str,
    ) -> None:
        output = result.model_dump(mode="json")
        output["deterministic_completeness"] = dict(checklist)
        self.logger.log_agent_action(
            agent_type="ComplianceOfficer",
            action="generate_narrative",
            case_id=case.case_id,
            input_data={
                "risk_analysis": risk_analysis.model_dump(mode="json"),
                "mode": mode,
            },
            output_data=output,
            reasoning=result.reasoning,
            execution_time_ms=(perf_counter() - started) * 1000,
            success=True,
            confidence_scores={"risk_analysis": risk_analysis.confidence_score},
            regulatory_flags=result.regulatory_citations,
        )

    @staticmethod
    def _extract_json_from_response(response_text: str) -> str:
        text = response_text.strip()
        if not text:
            raise ValueError("No JSON content found")

        decoder = json.JSONDecoder()
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                _, end = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            return text[index : index + end]
        raise ValueError("No JSON content found")

    @staticmethod
    def _format_transactions_for_compliance(
        transactions: Sequence[TransactionData],
    ) -> str:
        if not transactions:
            return "No transactions supplied"
        lines = []
        for index, transaction in enumerate(transactions, start=1):
            details = transaction.description or "No description"
            if transaction.location:
                details += f" at {transaction.location}"
            if transaction.method:
                details += f" via {transaction.method}"
            if transaction.counterparty:
                details += f" with {transaction.counterparty}"
            lines.append(
                f"{index}. {transaction.transaction_date.isoformat()}: "
                f"${transaction.amount:,.2f} {transaction.transaction_type}; {details}"
            )
        return "\n".join(lines)


__all__ = ["REGULATORY_SOURCES", "ComplianceOfficerAgent"]
