"""
Parallelization Pattern for Concurrent Fraud Detection

This module implements parallel fraud detection using multiple agents.
You will add a third fraud detection agent and implement aggregation.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import time
from agents.workflow_agents.base_agents import (
    FraudAmountDetectionAgent,
    FraudPatternDetectionAgent,
    GeographicRiskAgent,
    AIAnomalyDetectionAgent,
    FraudAggAgent
)


class ParallelizationPattern:
    """
    Implements parallel processing of fraud detection agents.
    Multiple agents analyze messages concurrently for better performance.
    """

    def __init__(self, max_workers: int = 8):
        """
        Initialize the parallelization pattern.

        Args:
            max_workers: Maximum number of concurrent threads
        """
        self.max_workers = max_workers

        # Initialize fraud detection agents
        # TODO 10: Create third agent (10 points)
        # INSTRUCTIONS:
        # 1. Add a third fraud detection agent to this list
        # 2. You can create a new agent class or use one of the existing ones
        # 3. Ideas for new agents:
        #    - BenfordLawAgent: Check if amounts follow Benford's Law
        #    - VelocityCheckAgent: Check transaction velocity/frequency
        #    - GeographicRiskAgent: Check sender/receiver country risk
        #    - TimeBasedRiskAgent: Check for unusual transaction times
        #
        # EXAMPLE:
        # If you create a new agent class, define it in base_agents.py first
        # Then add it here like:
        # from agents.workflow_agents.base_agents import YourNewAgent
        # list_of_agents = [
        #     FraudAmountDetectionAgent(),
        #     FraudPatternDetectionAgent(),
        #     YourNewAgent()  # <-- Add your third agent here
        # ]

        # TODO 10: The third agent is the AI-driven anomaly detector (our custom
        # fraud agent). GeographicRiskAgent is included as an additional
        # rule-based personality for richer, multi-agent screening.
        self.list_of_agents = [
            FraudAmountDetectionAgent(),
            FraudPatternDetectionAgent(),
            AIAnomalyDetectionAgent(),   # <-- third (custom, AI-driven) agent
            GeographicRiskAgent(),       # <-- fourth (jurisdiction risk) agent
        ]

        # Aggregator combines every agent's verdict into a single assessment.
        self.aggregator = FraudAggAgent()

    def _process_message(self, message: Dict, agent: Any) -> Dict:
        """
        Process a single message with a specific fraud detection agent.

        Args:
            message: SWIFT message to analyze
            agent: Fraud detection agent to use

        Returns:
            Fraud analysis results from the agent
        """
        try:
            # Call the agent's analyze method
            result = agent.analyze(message)
            result['message_id'] = message.get('message_id', 'unknown')
            return result
        except Exception as e:
            print(f"Error in agent {agent.__class__.__name__}: {e}")
            return {
                'agent': agent.__class__.__name__,
                'error': str(e),
                'risk_score': 0,
                'fraud_reasons': []
            }

    def process_batch_parallel(self, messages: List[Dict]) -> List[Dict]:
        """
        Process a batch of messages in parallel using all fraud detection agents.

        Args:
            messages: List of SWIFT messages to process

        Returns:
            List of messages with fraud detection results
        """
        print(f"Processing {len(messages)} messages with {len(self.list_of_agents)} agents in parallel...")
        start_time = time.time()

        # TODO 11: Aggregation agent that combines every fraud agent's verdict.
        aggregator = self.aggregator

        # Process messages in parallel
        processed_messages = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_msg = {}
            for message in messages:
                # Submit tasks for each agent to analyze this message
                agent_futures = []
                for agent in self.list_of_agents:
                    future = executor.submit(self._process_message, message, agent)
                    agent_futures.append(future)

                future_to_msg[message['message_id']] = {
                    'message': message,
                    'futures': agent_futures
                }

            # Collect results
            for msg_id, msg_data in future_to_msg.items():
                message = msg_data['message']
                agent_results = []

                # Wait for all agents to complete for this message
                for future in msg_data['futures']:
                    try:
                        result = future.result(timeout=5)
                        agent_results.append(result)
                    except Exception as e:
                        print(f"Error getting result for message {msg_id}: {e}")

                # TODO 12: Aggregate the agents' results and mark the message.
                if aggregator:
                    aggregated = aggregator.aggregate_results(agent_results)
                    message['fraud_status'] = (
                        "FRAUDULENT" if aggregated['is_fraudulent'] else "CLEAN"
                    )
                    message['fraud_score'] = aggregated['confidence']
                    message['fraud_reasons'] = aggregated['aggregated_reasons']
                else:
                    message['fraud_status'] = "PENDING"
                    message['fraud_score'] = 0
                    message['fraud_reasons'] = []

                # Keep the raw per-agent breakdown for reporting / audit.
                message['fraud_analysis'] = agent_results

                processed_messages.append(message)

        elapsed_time = time.time() - start_time
        print(f"Parallel processing completed in {elapsed_time:.2f} seconds")

        # Print fraud summary
        fraudulent_count = sum(1 for m in processed_messages
                              if m.get('fraud_status') == 'FRAUDULENT')
        print(f"Fraud Detection Summary: {fraudulent_count}/{len(processed_messages)} messages flagged as fraudulent")

        return processed_messages

    def test_agents(self):
        """
        Test method to verify all agents are working.
        You can use this to test your third agent.
        """
        test_message = {
            'message_id': 'TEST001',
            'amount': '15000.00 USD',
            'sender_bic': 'TESTUS33XXX',
            'receiver_bic': 'FAKEGB22XXX',
            'remittance_info': 'Urgent payment needed immediately'
        }

        print("Testing fraud detection agents:")
        print(f"Test message: {test_message}")
        print("\nAgent results:")

        for agent in self.list_of_agents:
            result = agent.analyze(test_message)
            print(f"\n{agent.__class__.__name__}:")
            print(f"  Risk Score: {result.get('risk_score', 0)}")
            print(f"  Reasons: {result.get('fraud_reasons', [])}")

        # Test aggregation if aggregator is available
        if hasattr(self, 'aggregator'):
            agent_results = [agent.analyze(test_message) for agent in self.list_of_agents]
            aggregated = self.aggregator.aggregate_results(agent_results)
            print(f"\nAggregated Result:")
            print(f"  Is Fraudulent: {aggregated['is_fraudulent']}")
            print(f"  Confidence: {aggregated['confidence']}%")
            print(f"  Total Risk Score: {aggregated['total_risk_score']}")


# Example of how to create a custom fraud detection agent
# You can use this as a template for TODO 10

class CustomFraudAgent:
    """
    Example template for creating a custom fraud detection agent.
    You can modify this for your third agent in TODO 10.
    """

    def __init__(self):
        # Initialize any rules or thresholds your agent needs
        self.threshold = 0.5

    def analyze(self, message: Dict) -> Dict:
        """
        Analyze a message for fraud indicators.

        Args:
            message: SWIFT message to analyze

        Returns:
            Dictionary with:
                - agent: Name of this agent
                - risk_score: Float between 0 and 1
                - fraud_reasons: List of reasons for the risk score
        """
        risk_score = 0
        fraud_reasons = []

        # Add your fraud detection logic here
        # Example: Check for specific patterns, amounts, or behaviors

        return {
            "agent": self.__class__.__name__,
            "risk_score": min(risk_score, 1.0),  # Keep between 0 and 1
            "fraud_reasons": fraud_reasons
        }


if __name__ == "__main__":
    # Test the parallelization pattern
    pattern = ParallelizationPattern()
    pattern.test_agents()