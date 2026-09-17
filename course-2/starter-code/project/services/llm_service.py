"""
LLM service for fraud analysis and SWIFT message correction using OpenAI.

All network access is routed through :mod:`services.llm_client`, which adds
retry-with-backoff, response caching and offline fallback. This means an
``LLMService`` can be constructed and used even when no API key is available –
the methods return conservative fallback results instead of crashing.
"""

import logging
from typing import Dict, List, Any

from models.swift_message import SWIFTMessage
from config import Config
from services import llm_client


class LLMService:
    """Service for LLM-based fraud analysis and SWIFT message correction."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        self.model = self.config.OPENAI_MODEL
        self.logger.info("LLM Service initialized with model: %s", self.model)

    def correct_swift_message(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Get a corrected SWIFT message from the LLM as a parsed dict."""
        return llm_client.chat_json(system_prompt, user_prompt, fallback={})

    def review_suspicious_transaction(self, message: SWIFTMessage, fraud_score: float,
                                      indicators: List[str]) -> Dict[str, Any]:
        """Use the LLM to review suspicious transactions and make a decision."""
        prompt = self._create_fraud_review_prompt(message, fraud_score, indicators)
        system = (
            "You are an expert fraud analyst specializing in SWIFT transactions. "
            "Analyze the provided transaction data and make a decision about whether to "
            "approve, reject, or hold the transaction for further investigation. "
            "Respond with JSON in the specified format."
        )

        def _fallback() -> Dict[str, Any]:
            # Conservative hold decision when the LLM is unavailable.
            return {
                "decision": "HOLD",
                "confidence": 0.5,
                "reasoning": "LLM unavailable; defaulting to conservative hold.",
                "risk_factors": indicators,
                "recommended_actions": ["Manual review required (LLM offline)"],
            }

        result = llm_client.chat_json(system, prompt, fallback_factory=_fallback)
        return result or _fallback()

    def get_swift_correction(self, prompt: str) -> Dict[str, Any]:
        """Get SWIFT message corrections from the LLM."""
        system = (
            "You are a SWIFT message validation expert. "
            "Your task is to correct SWIFT message format errors while "
            "maintaining the business intent of the transaction. "
            "Respond with JSON containing the corrected fields."
        )
        return llm_client.chat_json(system, prompt, fallback={})

    def analyze_benford_deviation(self, amounts: List[float], deviation_score: float,
                                  p_value: float) -> Dict[str, Any]:
        """Use the LLM to analyze Benford's Law deviations and provide insights."""
        prompt = self._create_benford_analysis_prompt(amounts, deviation_score, p_value)
        system = (
            "You are a financial forensics expert specializing in "
            "Benford's Law analysis for fraud detection. Analyze the provided "
            "transaction data and explain the significance of any deviations. "
            "Respond with JSON in the specified format."
        )

        def _fallback() -> Dict[str, Any]:
            return {
                "analysis": "Analysis unavailable (LLM offline).",
                "significance": "UNKNOWN",
                "recommendations": ["Manual review required"],
            }

        return llm_client.chat_json(system, prompt, fallback_factory=_fallback)

    def _create_fraud_review_prompt(self, message: SWIFTMessage, fraud_score: float,
                                    indicators: List[str]) -> str:
        """Create prompt for LLM fraud review."""
        prompt = f"""
Analyze the following SWIFT transaction for fraud risk:

TRANSACTION DETAILS:
- Message ID: {message.message_id}
- Type: {message.message_type}
- Reference: {message.reference}
- Amount: {message.amount} {message.currency}
- Sender BIC: {message.sender_bic}
- Receiver BIC: {message.receiver_bic}
- Value Date: {message.value_date}

AUTOMATED FRAUD ANALYSIS:
- Fraud Score: {fraud_score:.3f} (0.0 = no risk, 1.0 = high risk)
- Risk Indicators:
{chr(10).join(f"  - {indicator}" for indicator in indicators)}

ADDITIONAL CONTEXT:
- Ordering Customer: {getattr(message, 'ordering_customer', 'N/A')}
- Beneficiary: {getattr(message, 'beneficiary', 'N/A')}
- Remittance Info: {getattr(message, 'remittance_info', 'N/A')}

Based on this information, make a decision and provide analysis.

Respond with JSON in this exact format:
{{
    "decision": "APPROVE|HOLD|REJECT",
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation of your decision",
    "risk_factors": ["list", "of", "key", "risk", "factors"],
    "recommended_actions": ["list", "of", "recommended", "actions"],
    "business_impact": "Assessment of business impact if decision is wrong",
    "additional_checks": ["list", "of", "additional", "checks", "recommended"]
}}

Decision Guidelines:
- APPROVE: Low risk, process normally
- HOLD: Medium risk, requires manual review
- REJECT: High risk, block transaction
"""
        return prompt

    def _create_benford_analysis_prompt(self, amounts: List[float], deviation_score: float,
                                        p_value: float) -> str:
        """Create prompt for Benford's Law analysis."""
        first_digits = []
        for amount in amounts[:20]:
            amount_str = str(int(amount)).lstrip('0')
            if amount_str and amount_str[0].isdigit():
                first_digits.append(int(amount_str[0]))

        prompt = f"""
Analyze the following transaction data for Benford's Law compliance:

DATASET OVERVIEW:
- Total Transactions: {len(amounts)}
- Sample First Digits: {first_digits[:20]}
- Sample Amounts: {[f"${amt:,.2f}" for amt in amounts[:10]]}

STATISTICAL ANALYSIS:
- Deviation Score: {deviation_score:.4f}
- P-Value: {p_value:.6f}
- Significant Deviation: {p_value < 0.05}

BENFORD'S LAW CONTEXT:
Benford's Law states that in many real-world datasets, the first digit follows a specific distribution:
- Digit 1: ~30.1%
- Digit 2: ~17.6%
- Digit 3: ~12.5%
- etc.

Respond with JSON in this exact format:
{{
    "analysis": "Detailed analysis of the deviation and its implications",
    "significance": "LOW|MEDIUM|HIGH",
    "fraud_probability": 0.0-1.0,
    "likely_causes": ["list", "of", "likely", "causes"],
    "recommendations": ["list", "of", "recommended", "actions"],
    "false_positive_risk": "Assessment of false positive risk",
    "additional_analysis": "Suggestions for additional analysis"
}}
"""
        return prompt

    def batch_analyze_transactions(self, messages: List[SWIFTMessage]) -> Dict[str, Any]:
        """Perform batch analysis of multiple transactions for patterns."""
        try:
            amounts = [float(msg.amount) for msg in messages]
            currencies = [msg.currency for msg in messages]
            bics = [(msg.sender_bic, msg.receiver_bic) for msg in messages]

            prompt = f"""
Analyze this batch of {len(messages)} SWIFT transactions for suspicious patterns:

SUMMARY STATISTICS:
- Total Transactions: {len(messages)}
- Amount Range: ${min(amounts):,.2f} - ${max(amounts):,.2f}
- Average Amount: ${sum(amounts)/len(amounts):,.2f}
- Unique Currencies: {len(set(currencies))}
- Unique BIC Pairs: {len(set(bics))}

SAMPLE TRANSACTIONS:
{chr(10).join([
    f"- {msg.message_type} {msg.amount} {msg.currency} {msg.sender_bic}->{msg.receiver_bic}"
    for msg in messages[:10]
])}

Look for patterns that might indicate systematic fraud, money laundering,
structuring activities, or coordination between entities.
Respond with JSON format analysis of suspicious patterns found.
"""
            system = (
                "You are a financial crimes investigator analyzing "
                "transaction patterns for suspicious activity."
            )

            def _fallback() -> Dict[str, Any]:
                return {
                    "analysis": "Batch analysis unavailable (LLM offline).",
                    "patterns": [],
                    "recommendations": ["Manual review required"],
                }

            result = llm_client.chat_json(system, prompt, fallback_factory=_fallback)
            self.logger.info("LLM batch analysis completed")
            return result
        except Exception as exc:
            self.logger.error("LLM batch analysis failed: %s", exc)
            return {
                "analysis": "Batch analysis failed",
                "patterns": [],
                "recommendations": ["Manual review required"],
            }
