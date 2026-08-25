"""OpenAI-backed suspicious-activity classification with auditable evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from .foundation_sar import (
    AccountData,
    CaseData,
    ExplainabilityLogger,
    RiskAnalystOutput,
    TransactionData,
)


class RiskAnalystAgent:
    """Classify a case through one small, testable analysis interface."""

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
        self.system_prompt = """You are a Financial Crime Risk Analyst for an AML review system.
Use a five-step Chain-of-Thought framework, but return only a concise, auditable rationale rather than private hidden deliberation:
1. Data Review — summarize customer, account, transaction, and screening facts.
2. Pattern Recognition — identify temporal, amount, velocity, counterparty, channel, and geographic patterns.
3. Regulatory Mapping — map facts to relevant BSA/AML concepts without claiming a legal conclusion.
4. Risk Quantification — assign confidence from 0 to 1 and risk level Low, Medium, High, or Critical.
5. Classification Decision — choose exactly one: Structuring, Sanctions, Fraud, Money_Laundering, or Other.

Guardrails:
- Treat automated indicators as review evidence, never a filing decision.
- Never infer a sanctions match from country, name similarity, transaction text, or missing data. Classify Sanctions only when authoritative screening evidence explicitly confirms a match.
- Do not use transaction-ID labels as evidence.
- Distinguish observed facts from inference and acknowledge missing facts.
- Respond with one JSON object only using: classification, confidence_score, reasoning, key_indicators, risk_level.
"""

    def analyze_case(
        self,
        case: CaseData,
        screening_results: Sequence[Mapping[str, Any]] | None = None,
    ) -> RiskAnalystOutput:
        """Analyze one validated case, using deterministic fallback when no client exists."""

        if not isinstance(case, CaseData):
            raise TypeError("case must be a CaseData instance")

        started = perf_counter()
        evidence = self.extract_case_evidence(case, screening_results)
        if self.client is None:
            result = self._deterministic_analysis(case, evidence)
            self._log_success(case, result, evidence, started, mode="deterministic")
            return result

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.3,
                max_tokens=1000,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self._build_user_prompt(case, evidence)},
                ],
            )
            content = response.choices[0].message.content
        except Exception as exc:
            elapsed_ms = (perf_counter() - started) * 1000
            self.logger.log_agent_action(
                agent_type="RiskAnalyst",
                action="analyze_case",
                case_id=case.case_id,
                input_data={"evidence": evidence, "mode": "openai"},
                reasoning=f"API request failed: {exc}",
                execution_time_ms=elapsed_ms,
                success=False,
                error_message=str(exc),
            )
            if self.fallback_on_error:
                result = self._deterministic_analysis(case, evidence)
                self._log_success(
                    case,
                    result,
                    evidence,
                    started,
                    mode="deterministic_fallback",
                )
                return result
            raise RuntimeError(f"Risk Analyst API request failed: {exc}") from exc

        try:
            json_text = self._extract_json_from_response(content or "")
            payload = json.loads(json_text)
            payload["case_id"] = case.case_id
            result = RiskAnalystOutput.model_validate(payload)
        except (ValueError, TypeError, json.JSONDecodeError, ValidationError) as exc:
            elapsed_ms = (perf_counter() - started) * 1000
            self.logger.log_agent_action(
                agent_type="RiskAnalyst",
                action="analyze_case",
                case_id=case.case_id,
                input_data={"evidence": evidence, "mode": "openai"},
                reasoning=f"JSON parsing failed: {exc}",
                execution_time_ms=elapsed_ms,
                success=False,
                error_message=str(exc),
            )
            if self.fallback_on_error:
                result = self._deterministic_analysis(case, evidence)
                self._log_success(
                    case,
                    result,
                    evidence,
                    started,
                    mode="deterministic_fallback",
                )
                return result
            raise ValueError(f"Failed to parse Risk Analyst JSON output: {exc}") from exc

        self._log_success(case, result, evidence, started, mode="openai")
        return result

    def extract_case_evidence(
        self,
        case: CaseData,
        screening_results: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Compute deterministic facts once so prompts and logs use identical evidence."""

        transactions = case.transactions
        amounts = [transaction.amount for transaction in transactions]
        dates = [transaction.transaction_date for transaction in transactions]
        cash_below_ctr = [
            transaction
            for transaction in transactions
            if transaction.transaction_type == "Cash_Deposit"
            and 9_000 <= transaction.amount < 10_000
        ]
        wires = [
            transaction
            for transaction in transactions
            if "wire" in transaction.transaction_type.lower()
            or (transaction.method and "wire" in transaction.method.lower())
        ]
        counterparties = {
            transaction.counterparty
            for transaction in transactions
            if transaction.counterparty
        }
        locations = {
            transaction.location for transaction in transactions if transaction.location
        }
        confirmed_sanctions = [
            dict(result)
            for result in screening_results or ()
            if result.get("source") == "OFAC_SLS"
            and result.get("match_status") == "confirmed"
        ]
        return {
            "transaction_count": len(transactions),
            "total_amount": round(sum(amounts), 2),
            "largest_transaction": round(max(amounts, default=0), 2),
            "activity_start": min(dates).isoformat() if dates else None,
            "activity_end": max(dates).isoformat() if dates else None,
            "cash_deposits_9000_to_under_10000": len(cash_below_ctr),
            "cash_band_total": round(sum(item.amount for item in cash_below_ctr), 2),
            "wire_count": len(wires),
            "wire_total": round(sum(item.amount for item in wires), 2),
            "unique_counterparties": len(counterparties),
            "unique_locations": len(locations),
            "confirmed_ofac_matches": confirmed_sanctions,
            "missing_counterparty_count": sum(
                1 for transaction in transactions if not transaction.counterparty
            ),
            "missing_location_count": sum(
                1 for transaction in transactions if not transaction.location
            ),
        }

    def _deterministic_analysis(
        self, case: CaseData, evidence: Mapping[str, Any]
    ) -> RiskAnalystOutput:
        indicators: list[str] = []
        if evidence["confirmed_ofac_matches"]:
            classification = "Sanctions"
            risk_level = "Critical"
            confidence = 0.98
            indicators.append("Authoritative OFAC SLS screening result marked confirmed")
        elif evidence["cash_deposits_9000_to_under_10000"] >= 3:
            classification = "Structuring"
            risk_level = "High"
            confidence = min(
                0.98,
                0.65 + 0.04 * evidence["cash_deposits_9000_to_under_10000"],
            )
            indicators.extend(
                [
                    f"{evidence['cash_deposits_9000_to_under_10000']} cash deposits from $9,000 to under $10,000",
                    f"Cash-band total ${evidence['cash_band_total']:,.2f}",
                ]
            )
        elif any(
            token in (transaction.description or "").lower()
            for transaction in case.transactions
            for token in ("unauthorized", "identity theft", "account takeover", "fraud")
        ):
            classification = "Fraud"
            risk_level = "High"
            confidence = 0.82
            indicators.append("Transaction descriptions contain explicit fraud-related facts")
        elif evidence["wire_count"] >= 3 and evidence["wire_total"] >= 50_000:
            classification = "Money_Laundering"
            risk_level = "High"
            confidence = 0.72
            indicators.extend(
                [
                    f"{evidence['wire_count']} wire transactions",
                    f"Wire total ${evidence['wire_total']:,.2f}",
                    f"{evidence['unique_counterparties']} identified counterparties",
                ]
            )
        else:
            classification = "Other"
            risk_level = "Low" if evidence["transaction_count"] < 20 else "Medium"
            confidence = 0.55
            indicators.append("No supported primary classification threshold was met")

        reasoning = (
            f"Data Review: {evidence['transaction_count']} transactions totaling "
            f"${evidence['total_amount']:,.2f}. Pattern Recognition: "
            f"{'; '.join(indicators)}. Regulatory Mapping: indicators are triage "
            "evidence and require human review. Risk Quantification: "
            f"{risk_level} at {confidence:.2f} confidence. Classification Decision: "
            f"{classification}."
        )
        return RiskAnalystOutput(
            case_id=case.case_id,
            suspicious_activity_type=classification,
            confidence_score=confidence,
            risk_level=risk_level,
            reasoning=reasoning,
            suspicious_indicators=indicators,
        )

    def _build_user_prompt(
        self, case: CaseData, evidence: Mapping[str, Any]
    ) -> str:
        customer = case.customer
        return (
            "Analyze this validated case without relying on transaction-ID labels.\n"
            f"Case: {case.case_id}\n"
            f"Customer: {customer.name} ({customer.customer_id}); "
            f"customer risk rating: {customer.risk_rating}; "
            f"occupation: {customer.occupation or 'not provided'}; "
            f"annual income: {customer.annual_income or 'not provided'}\n"
            f"Accounts:\n{self._format_accounts(case.accounts)}\n"
            f"Transactions:\n{self._format_transactions(case.transactions)}\n"
            f"Computed evidence:\n{json.dumps(evidence, sort_keys=True)}"
        )

    def _log_success(
        self,
        case: CaseData,
        result: RiskAnalystOutput,
        evidence: Mapping[str, Any],
        started: float,
        *,
        mode: str,
    ) -> None:
        self.logger.log_agent_action(
            agent_type="RiskAnalyst",
            action="analyze_case",
            case_id=case.case_id,
            input_data={"evidence": dict(evidence), "mode": mode},
            output_data=result.model_dump(mode="json"),
            reasoning=result.reasoning,
            execution_time_ms=(perf_counter() - started) * 1000,
            success=True,
            confidence_scores={"classification": result.confidence_score},
            regulatory_flags=result.suspicious_indicators,
        )

    @staticmethod
    def _extract_json_from_response(response_text: str) -> str:
        """Return the first complete JSON object from plain text or code fences."""

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
    def _format_accounts(accounts: Sequence[AccountData]) -> str:
        if not accounts:
            return "No accounts supplied"
        return "\n".join(
            f"- {account.account_id}: {account.account_type}; "
            f"balance ${account.current_balance:,.2f}; average monthly "
            f"${account.average_monthly_balance:,.2f}; status {account.status}"
            for account in accounts
        )

    @staticmethod
    def _format_transactions(transactions: Sequence[TransactionData]) -> str:
        if not transactions:
            return "No transactions supplied"

        def render_date(value: date) -> str:
            return value.isoformat()

        lines = []
        for index, transaction in enumerate(transactions, start=1):
            details = [transaction.description or "No description"]
            if transaction.counterparty:
                details.append(f"counterparty {transaction.counterparty}")
            if transaction.location:
                details.append(f"location {transaction.location}")
            lines.append(
                f"{index}. {render_date(transaction.transaction_date)}: "
                f"{transaction.transaction_type} ${transaction.amount:,.2f}; "
                + "; ".join(details)
            )
        return "\n".join(lines)


__all__ = ["RiskAnalystAgent"]
