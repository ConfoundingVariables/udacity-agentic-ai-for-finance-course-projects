"""
Prompt Chaining Pattern for Multi-Stage Fraud Analysis

This module implements a chain of specialized agents for comprehensive fraud analysis.
Each agent in the chain builds upon the insights from previous agents.
"""

import json
from typing import Dict, List, Any
from config import Config
from services import llm_client


class PromptChainingPattern:
    """
    Implements a chain of agents for progressive fraud analysis.
    Each agent adds your specialized analysis to build a comprehensive assessment.
    """

    def __init__(self):
        """Initialize the prompt chaining pattern."""
        self.config = Config()
        self.model = Config.OPENAI_MODEL
        self.temperature = Config.LLM_TEMPERATURE  # Low temp for consistency

    def _create_initial_screener_prompt(self, messages: List[Dict]) -> tuple:
        """
        Create prompt for the initial screening agent.

        Args:
            messages: List of SWIFT messages to screen

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = """You are an Initial Fraud Screener specializing in rapid triage of SWIFT transactions.
        Your role is to quickly categorize transactions into risk levels.

        For each transaction, assign:
        - GREEN: Low risk, standard processing
        - YELLOW: Medium risk, needs review
        - RED: High risk, immediate attention

        Consider: amounts, BIC codes, countries, and obvious red flags.
        Return your analysis in JSON format."""

        user_prompt = f"""Perform initial screening on these SWIFT messages:
        {json.dumps(messages, indent=2)}

        Return JSON with structure:
        {{
            "screening_results": [
                {{
                    "message_id": "...",
                    "risk_level": "GREEN|YELLOW|RED",
                    "initial_flags": ["..."],
                    "recommended_action": "..."
                }}
            ],
            "summary": "Overall batch assessment"
        }}"""

        return system_prompt, user_prompt

    def _create_technical_analyst_prompt(self, messages: List[Dict], initial_screening: Dict) -> tuple:
        """
        Create prompt for the technical analyst agent.

        Args:
            messages: Original SWIFT messages
            initial_screening: Results from initial screener

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = """You are a Technical Analyst specializing in SWIFT message format validation.
        Review the initial screening results and perform deep technical analysis.

        Focus on:
        - SWIFT format compliance (MT103/MT202 standards)
        - BIC code validation and legitimacy
        - Amount format and currency validation
        - Reference number patterns
        - Date format compliance

        Build upon the initial screening to identify technical anomalies.
        Return your analysis in JSON format."""

        user_prompt = f"""Review these SWIFT messages with initial screening results:

        Messages: {json.dumps(messages, indent=2)}

        Initial Screening: {json.dumps(initial_screening, indent=2)}

        Perform technical validation and return JSON with:
        {{
            "technical_analysis": [
                {{
                    "message_id": "...",
                    "format_compliance": true/false,
                    "bic_validation": {{"sender": "status", "receiver": "status"}},
                    "technical_issues": ["..."],
                    "risk_adjustment": "increase|maintain|decrease",
                    "technical_score": 0-100
                }}
            ],
            "technical_summary": "Overall technical assessment"
        }}"""

        return system_prompt, user_prompt

    def _create_compliance_officer_prompt(self, messages: List[Dict], chain_results: Dict) -> tuple:
        """
        Create prompt for the compliance officer agent.

        Args:
            messages: Original SWIFT messages
            chain_results: All previous analysis results

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = """You are a Compliance Officer specializing in AML and regulatory compliance.
        Review all previous analysis and assess regulatory compliance risks.

        Focus on:
        - AML (Anti-Money Laundering) red flags
        - Sanctions screening indicators
        - PEP (Politically Exposed Persons) risks
        - Regulatory reporting requirements
        - KYC (Know Your Customer) concerns

        Consider the complete analysis chain to make compliance determinations.
        Return your analysis in JSON format."""

        user_prompt = f"""Review these SWIFT messages with complete analysis chain:

        Messages: {json.dumps(messages, indent=2)}

        Analysis Chain Results: {json.dumps(chain_results, indent=2)}

        Perform compliance assessment and return JSON with:
        {{
            "compliance_review": [
                {{
                    "message_id": "...",
                    "aml_risk": "low|medium|high",
                    "sanctions_risk": "clear|potential|confirmed",
                    "compliance_issues": ["..."],
                    "required_actions": ["..."],
                    "compliance_score": 0-100
                }}
            ],
            "compliance_summary": "Overall compliance assessment",
            "escalation_required": true/false
        }}"""

        return system_prompt, user_prompt

    def _create_final_reviewer_prompt(self, messages: List[Dict], complete_chain: Dict) -> tuple:
        """
        Create prompt for the final review agent.

        Args:
            messages: Original SWIFT messages
            complete_chain: All analysis results from the chain

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = """You are the Final Reviewer responsible for synthesizing all analysis.
        Make the final fraud determination based on the complete analysis chain.

        Review all findings from:
        1. Initial Screening
        2. Technical Analysis
        3. Risk Assessment (if completed)
        4. Compliance Review

        Make final decisions: APPROVE, HOLD, or REJECT each transaction.
        Provide clear justification based on the accumulated evidence.
        Return your analysis in JSON format."""

        user_prompt = f"""Make final determinations based on complete analysis:

        Messages: {json.dumps(messages, indent=2)}

        Complete Analysis Chain: {json.dumps(complete_chain, indent=2)}

        Return final decisions in JSON:
        {{
            "final_decisions": [
                {{
                    "message_id": "...",
                    "decision": "APPROVE|HOLD|REJECT",
                    "confidence": 0-100,
                    "key_factors": ["..."],
                    "justification": "...",
                    "follow_up_required": ["..."]
                }}
            ],
            "batch_summary": {{
                "approved": 0,
                "held": 0,
                "rejected": 0,
                "overall_risk": "low|medium|high"
            }}
        }}"""

        return system_prompt, user_prompt

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Dict:
        """
        Make a call to the LLM with the given prompts.

        Args:
            system_prompt: System role prompt
            user_prompt: User message prompt

        Returns:
            Parsed JSON response from the LLM (retry, caching and offline
            fallback are handled by the shared llm_client).
        """
        return llm_client.chat_json(
            system_prompt,
            user_prompt,
            model=self.model,
            temperature=self.temperature,
            fallback={},
        )

    def process_chain(self, messages: List[Dict]) -> Dict:
        """
        Process messages through the complete prompt chain.

        Args:
            messages: List of SWIFT messages to analyze

        Returns:
            Complete analysis results from all agents in the chain
        """
        print("Starting Prompt Chaining Analysis...")
        chain_results = {}

        # Step 1: Initial Screening
        print("Step 1: Initial Screener analyzing messages...")
        system_prompt, user_prompt = self._create_initial_screener_prompt(messages)
        initial_results = self._call_llm(system_prompt, user_prompt)
        chain_results['initial_screening'] = initial_results

        # TODO 13: Step 2 - Technical Analyst (deep SWIFT format validation).
        print("Step 2: Technical Analyst reviewing messages...")
        system_prompt, user_prompt = self._create_technical_analyst_prompt(
            messages, initial_results
        )
        technical_results = self._call_llm(system_prompt, user_prompt)
        chain_results['technical_analysis'] = technical_results

        # Step 3: Risk Assessor (Provided as example)
        print("Step 3: Risk Assessor evaluating patterns...")
        # This step is implemented for you as an example
        risk_prompt_system = """You are a Risk Assessment Specialist.
        Analyze behavioral patterns and transaction risks based on previous findings.
        Focus on velocity, patterns, and behavioral anomalies."""

        risk_prompt_user = f"""Assess risk based on analysis so far:
        Messages: {json.dumps(messages, indent=2)}
        Current Analysis: {json.dumps(chain_results, indent=2)}

        Return JSON with risk scores and pattern analysis."""

        risk_results = self._call_llm(risk_prompt_system, risk_prompt_user)
        chain_results['risk_assessment'] = risk_results

        # TODO 14: Step 4 - Compliance Officer (AML / sanctions / KYC review).
        print("Step 4: Compliance Officer reviewing for regulatory issues...")
        system_prompt, user_prompt = self._create_compliance_officer_prompt(
            messages, chain_results
        )
        compliance_results = self._call_llm(system_prompt, user_prompt)
        chain_results['compliance_review'] = compliance_results

        # Step 5: Final Reviewer (Provided)
        print("Step 5: Final Reviewer making decisions...")
        system_prompt, user_prompt = self._create_final_reviewer_prompt(messages, chain_results)
        final_results = self._call_llm(system_prompt, user_prompt)
        chain_results['final_review'] = final_results

        # Update messages with final decisions
        if 'final_decisions' in final_results:
            decision_map = {d['message_id']: d for d in final_results['final_decisions']}
            for message in messages:
                msg_id = message.get('message_id')
                if msg_id in decision_map:
                    decision = decision_map[msg_id]
                    message['fraud_decision'] = decision['decision']
                    message['fraud_confidence'] = decision.get('confidence', 0)
                    message['fraud_justification'] = decision.get('justification', '')

        print("Prompt Chaining Analysis Complete!")
        return chain_results

    def test_chain(self):
        """
        Test the prompt chain with sample messages.
        You can use this to verify your implementations.
        """
        test_messages = [
            {
                'message_id': 'TEST001',
                'message_type': 'MT103',
                'amount': '50000.00 USD',
                'sender_bic': 'CHASUS33XXX',
                'receiver_bic': 'DEUTGB22XXX',
                'reference': 'REF123456',
                'remittance_info': 'Invoice payment for services'
            },
            {
                'message_id': 'TEST002',
                'message_type': 'MT103',
                'amount': '999999.99 USD',
                'sender_bic': 'TESTUS33XXX',
                'receiver_bic': 'FAKEGB22XXX',
                'reference': 'URGENT999',
                'remittance_info': 'Urgent confidential transfer'
            }
        ]

        print("Testing Prompt Chain with sample messages:")
        results = self.process_chain(test_messages)

        # Print summary of results
        print("\n=== Chain Results Summary ===")
        for stage, data in results.items():
            print(f"\n{stage.upper()}:")
            if isinstance(data, dict):
                # Print key findings from each stage
                if 'summary' in data:
                    print(f"  Summary: {data['summary']}")
                elif 'batch_summary' in data:
                    print(f"  Batch Summary: {data['batch_summary']}")

        return results


if __name__ == "__main__":
    # Test the prompt chaining pattern
    # Note: Requires OPENAI_API_KEY environment variable
    pattern = PromptChainingPattern()
    pattern.test_chain()