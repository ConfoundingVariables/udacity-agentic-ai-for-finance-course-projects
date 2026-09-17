"""
Base Agent Classes for SWIFT Transaction Processing.

This module contains the base classes that all agents inherit from, the
SWIFT correction agent used by the evaluator-optimizer, and the family of
fraud-detection "personalities" used by the parallelization pattern:

    * FraudAmountDetectionAgent   - rule-based, amount heuristics
    * FraudPatternDetectionAgent  - rule-based, BIC / keyword patterns
    * GeographicRiskAgent         - rule-based, jurisdiction risk
    * AIAnomalyDetectionAgent     - AI-driven anomaly detection (custom agent)
    * FraudAggAgent               - aggregates the above into one verdict
"""

import json
import math
from abc import ABC, abstractmethod
from typing import Dict

from config import Config
from services.llm_service import LLMService
from services import llm_client
from services.llm_client import get_client, chat_json


# --------------------------------------------------------------------------- #
# TODO 6: BaseAgent abstract class
# --------------------------------------------------------------------------- #
class BaseAgent(ABC):
    """Abstract base class shared by all LLM-backed agents.

    Concrete agents implement :meth:`create_prompt` to describe how their input
    data becomes a user prompt, and rely on :meth:`respond` for the common LLM
    call (which already provides retry, caching and offline fallback).
    """

    def __init__(self, name: str = None, system_prompt: str = None):
        self.config = Config()
        # Shared LLM integration service (safe to construct even with no key).
        self.llm_service = LLMService()
        self.name = name or self.__class__.__name__
        self.system_prompt = system_prompt or (
            "You are a helpful assistant supporting SWIFT transaction processing. "
            "Always respond with valid JSON."
        )

    @abstractmethod
    def create_prompt(self, data) -> str:
        """Each agent must implement its own prompt creation and return a str."""
        raise NotImplementedError

    def respond(self, prompt: str) -> Dict:
        """Common method to get a parsed JSON response from the LLM."""
        return chat_json(self.system_prompt, prompt, fallback={})


class SwiftCorrectionAgent:
    """Agent for correcting SWIFT messages based on validation errors."""

    def __init__(self):
        # TODO 7: Define LLMService.
        self.llm_service = LLMService()

    def create_prompt(self, message, errors):
        """Create a (system_prompt, user_prompt) pair to correct a SWIFT message."""
        system_prompt = """You are a SWIFT message correction expert.
        Fix the validation errors while maintaining the business intent.
        Return the corrected message in JSON format."""

        user_prompt = f"""
        Original SWIFT Message:
        {message}

        Validation Errors to Fix:
        {errors}

        Please correct these errors and return the complete corrected message in JSON format.
        """

        return system_prompt, user_prompt

    def respond(self, message, errors):
        """Get an LLM response to correct the SWIFT message.

        Returns the corrected message dict, or the original message if the
        correction call fails.
        """
        system_prompt, user_prompt = self.create_prompt(message, errors)

        if not Config.has_api_key():
            # Offline: no correction is possible, keep the original message.
            return message

        try:
            # Use the shared, pre-configured client (honours the Vocareum base
            # URL and API key from Config).
            client = get_client()
            request = {
                "model": Config.OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": Config.LLM_TEMPERATURE,
            }
            # TODO 8: Set response format so the model returns valid JSON
            # (only when the endpoint supports it; some models reject it).
            if llm_client._JSON_MODE_SUPPORTED:
                request["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**request)

            # TODO 9: Parse the JSON content from the response.
            content = response.choices[0].message.content
            result = llm_client._parse_json_content(content or "{}")
            return result or message

        except Exception as e:
            print(f"Error in SwiftCorrectionAgent: {e}")
            return message  # Return original if correction fails


class FraudAmountDetectionAgent:
    """Agent for detecting fraud based on transaction amounts."""

    def __init__(self):
        self.rules = [
            {"condition": "amount > 10000", "risk_score": 0.3},
            {"condition": "round_amount", "risk_score": 0.2},
            {"condition": "unusual_precision", "risk_score": 0.1},
        ]

    def analyze(self, message):
        """Analyze a SWIFT message for amount-based fraud patterns."""
        risk_score = 0
        fraud_reasons = []

        try:
            # Amount may be a string ("1000.00") or a number (after an LLM
            # correction); coerce to str before scanning characters.
            amount_str = str(message.get('amount', '0'))
            amount = float(''.join(c for c in amount_str if c.isdigit() or c == '.'))

            # Rule 1: Large amounts
            if amount > 10000:
                risk_score += 0.3
                fraud_reasons.append(f"High amount transaction: {amount}")

            # Rule 2: Round amounts (multiples of 1000)
            if amount % 1000 == 0 and amount > 0:
                risk_score += 0.2
                fraud_reasons.append(f"Suspiciously round amount: {amount}")

            # Rule 3: Unusual precision for large amounts
            if amount > 100000 and (amount % 1) != 0:
                risk_score += 0.1
                fraud_reasons.append("Large amount with unusual decimal precision")

        except (ValueError, TypeError) as e:
            print(f"Error analyzing amount: {e}")

        return {
            "agent": "FraudAmountDetectionAgent",
            "risk_score": min(risk_score, 1.0),
            "fraud_reasons": fraud_reasons,
        }


class FraudPatternDetectionAgent:
    """Agent for detecting fraud based on transaction patterns."""

    def __init__(self):
        self.high_risk_patterns = ['TEST', 'FAKE', 'DEMO', '999', '000000']
        self.suspicious_keywords = ['urgent', 'immediately', 'secret', 'confidential']

    def analyze(self, message):
        """Analyze a SWIFT message for pattern-based fraud indicators."""
        risk_score = 0
        fraud_reasons = []

        sender_bic = message.get('sender_bic', '')
        receiver_bic = message.get('receiver_bic', '')

        for pattern in self.high_risk_patterns:
            if pattern in sender_bic.upper() or pattern in receiver_bic.upper():
                risk_score += 0.4
                fraud_reasons.append(f"Test/fake pattern detected in BIC: {pattern}")

        if sender_bic and sender_bic == receiver_bic:
            risk_score += 0.5
            fraud_reasons.append("Same sender and receiver BIC")

        remittance = message.get('remittance_info', '') or ''
        remittance = remittance.lower()
        for keyword in self.suspicious_keywords:
            if keyword in remittance:
                risk_score += 0.2
                fraud_reasons.append(f"Suspicious keyword in remittance: {keyword}")

        return {
            "agent": "FraudPatternDetectionAgent",
            "risk_score": min(risk_score, 1.0),
            "fraud_reasons": fraud_reasons,
        }


class GeographicRiskAgent:
    """Rule-based agent scoring jurisdiction risk from BIC country codes.

    The 5th-6th characters of a BIC encode the ISO country code. This agent
    flags transfers touching higher-risk jurisdictions and cross-border /
    unusual-corridor transfers.
    """

    def __init__(self):
        # Illustrative high-risk jurisdiction list for the exercise.
        self.high_risk_countries = {"IR", "KP", "SY", "RU", "AF", "MM", "VE"}
        self.medium_risk_countries = {"KY", "PA", "BS", "VG", "SC", "LB"}

    @staticmethod
    def _country_from_bic(bic: str) -> str:
        return bic[4:6].upper() if bic and len(bic) >= 6 else ""

    def analyze(self, message: Dict) -> Dict:
        risk_score = 0.0
        fraud_reasons = []

        sender_country = self._country_from_bic(message.get('sender_bic', ''))
        receiver_country = self._country_from_bic(message.get('receiver_bic', ''))

        for label, country in (("sender", sender_country), ("receiver", receiver_country)):
            if country in self.high_risk_countries:
                risk_score += 0.5
                fraud_reasons.append(f"High-risk {label} jurisdiction: {country}")
            elif country in self.medium_risk_countries:
                risk_score += 0.25
                fraud_reasons.append(f"Medium-risk {label} jurisdiction: {country}")

        # Cross-border adds a small amount of inherent risk.
        if sender_country and receiver_country and sender_country != receiver_country:
            risk_score += 0.1
            fraud_reasons.append(
                f"Cross-border corridor: {sender_country} -> {receiver_country}"
            )

        return {
            "agent": "GeographicRiskAgent",
            "risk_score": min(risk_score, 1.0),
            "fraud_reasons": fraud_reasons,
        }


class AIAnomalyDetectionAgent(BaseAgent):
    """AI-driven anomaly detection agent (the custom fraud agent, TODO 10).

    Strategy
    --------
    1. Compute a cheap statistical anomaly signal (Benford first-digit rarity,
       round-number bias, extreme magnitude).
    2. Only escalate *borderline* transactions to the LLM for a nuanced
       anomaly judgement – a deliberate performance optimization that avoids an
       API call for obviously-clean traffic.
    3. Degrade gracefully to the statistical signal when the LLM is offline.
    """

    # Benford's Law expected first-digit frequencies.
    BENFORD = {d: math.log10(1 + 1 / d) for d in range(1, 10)}
    # Escalate to the LLM only when the heuristic is in a suspicious band.
    ESCALATION_THRESHOLD = 0.25

    def __init__(self):
        super().__init__(
            name="AIAnomalyDetectionAgent",
            system_prompt=(
                "You are an AI anomaly-detection specialist for SWIFT payments. "
                "Given a single transaction and precomputed statistical signals, "
                "judge how anomalous it is. Respond ONLY with JSON: "
                '{"anomaly_score": 0.0-1.0, "is_anomalous": true/false, '
                '"reasons": ["..."]}'
            ),
        )

    def create_prompt(self, data) -> str:
        """Build the user prompt from a message + heuristic signal dict."""
        message = data["message"]
        signals = data["signals"]
        return (
            "Assess this SWIFT transaction for anomalies.\n\n"
            f"Transaction: {json.dumps(message, default=str)}\n\n"
            f"Statistical signals: {json.dumps(signals)}\n\n"
            'Return JSON: {"anomaly_score": 0.0-1.0, "is_anomalous": true/false, '
            '"reasons": ["short reason", ...]}'
        )

    def _heuristic(self, message: Dict) -> Dict:
        """Cheap statistical anomaly signal in [0, 1] with reasons."""
        score = 0.0
        reasons = []
        try:
            amount_str = message.get('amount', '0')
            amount = float(''.join(c for c in str(amount_str) if c.isdigit() or c == '.'))
        except (ValueError, TypeError):
            amount = 0.0

        if amount > 0:
            first_digit = int(str(int(amount)).lstrip('0')[0]) if int(amount) > 0 else 1
            expected = self.BENFORD.get(first_digit, 0.05)
            # Rarer-than-expected leading digits contribute to the signal.
            if expected < 0.10:
                score += 0.2
                reasons.append(f"Rare leading digit {first_digit} (Benford p={expected:.2f})")

            if amount % 1000 == 0:
                score += 0.15
                reasons.append("Perfectly round amount")

            if amount > 1_000_000:
                score += 0.25
                reasons.append(f"Very large amount: {amount:,.2f}")
            elif amount > 250_000:
                score += 0.1
                reasons.append(f"Large amount: {amount:,.2f}")

        return {"heuristic_score": round(min(score, 1.0), 3), "reasons": reasons}

    def analyze(self, message: Dict) -> Dict:
        signal = self._heuristic(message)
        heuristic_score = signal["heuristic_score"]
        reasons = list(signal["reasons"])
        ai_used = False

        # Selective escalation: only ask the LLM about borderline cases.
        if heuristic_score >= self.ESCALATION_THRESHOLD and Config.has_api_key():
            prompt = self.create_prompt({"message": message, "signals": signal})
            ai_result = self.respond(prompt)  # retry + cache + fallback built in
            if ai_result:
                ai_used = True
                ai_score = float(ai_result.get("anomaly_score", heuristic_score) or 0.0)
                ai_score = max(0.0, min(ai_score, 1.0))
                # Blend heuristic and AI judgement (AI weighted higher).
                blended = 0.4 * heuristic_score + 0.6 * ai_score
                for r in ai_result.get("reasons", [])[:3]:
                    reasons.append(f"AI: {r}")
                final_score = blended
            else:
                final_score = heuristic_score
        else:
            final_score = heuristic_score

        return {
            "agent": "AIAnomalyDetectionAgent",
            "risk_score": round(min(final_score, 1.0), 3),
            "fraud_reasons": reasons,
            "ai_escalated": ai_used,
        }


class FraudAggAgent:
    """Agent for aggregating fraud detection results from multiple agents."""

    def __init__(self):
        self.threshold = 0.5  # Fraud threshold (50%)

    def aggregate_results(self, fraud_results):
        """Aggregate fraud detection results from multiple agents."""
        if not fraud_results:
            return {
                "is_fraudulent": False,
                "confidence": 0,
                "total_risk_score": 0,
                "aggregated_reasons": [],
            }

        total_risk = sum(r.get('risk_score', 0) for r in fraud_results)
        avg_risk = total_risk / len(fraud_results)

        all_reasons = []
        for result in fraud_results:
            agent_name = result.get('agent', 'Unknown')
            for reason in result.get('fraud_reasons', []):
                all_reasons.append(f"[{agent_name}] {reason}")

        is_fraudulent = avg_risk >= self.threshold

        return {
            "is_fraudulent": is_fraudulent,
            "confidence": round(avg_risk * 100, 2),
            "total_risk_score": round(avg_risk, 3),
            "aggregated_reasons": all_reasons,
        }


if __name__ == "__main__":
    # Quick smoke test of the (offline-safe) fraud agents.
    sample = {
        'message_id': 'TEST001',
        'amount': '9000000.00 USD',
        'sender_bic': 'TESTUS33XXX',
        'receiver_bic': 'FAKERU22XXX',
        'remittance_info': 'Urgent confidential transfer',
    }
    agents = [
        FraudAmountDetectionAgent(),
        FraudPatternDetectionAgent(),
        GeographicRiskAgent(),
        AIAnomalyDetectionAgent(),
    ]
    results = [a.analyze(sample) for a in agents]
    for r in results:
        print(r)
    print("AGGREGATE:", FraudAggAgent().aggregate_results(results))
