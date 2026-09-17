"""
SWIFT Transaction Processing System with Agent Patterns
Main application entry point.

This is the main integration point where all agent patterns work together
to process SWIFT messages through a complete pipeline:

    1. Evaluator-Optimizer  -> validate & correct messages
    2. Parallelization      -> concurrent multi-agent fraud screening
    3. Prompt Chaining      -> deep multi-stage fraud investigation
    4. Orchestrator-Worker  -> delegate final processing to capable workers

It then produces TWO different report sets and computes the STP rate.
"""

import logging
from typing import List, Dict

from config import Config
from services.swift_generator import SWIFTGenerator
from services.reporting import ReportGenerator

# Import the agent patterns you'll be using
from agents.evaluator_optimizer import EvaluatorOptimizerPattern
from agents.parallelization import ParallelizationPattern
from agents.orchestrator_worker import OrchestratorWorkerPattern
from agents.prompt_chaining import PromptChainingPattern

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class SWIFTProcessingSystem:
    """Main system orchestrating all agent patterns for SWIFT processing."""

    # Named filters used to produce different report sets (TODO 5).
    FILTERS = {
        "non_fraudulent": lambda m: m.get('fraud_status') != "FRAUDULENT",
        "fraudulent": lambda m: m.get('fraud_status') == "FRAUDULENT",
        "high_value": lambda m: _amount_of(m) > 50000,
        "mt103_only": lambda m: m.get('message_type') == "MT103",
        "mt202_only": lambda m: m.get('message_type') == "MT202",
    }

    def __init__(self):
        self.config = Config()
        self.swift_generator = SWIFTGenerator()
        self.report_generator = ReportGenerator()

        # Initialize agent patterns
        self.evaluator_optimizer = EvaluatorOptimizerPattern()
        self.parallelization_agent = ParallelizationPattern()
        self.orchestrator_worker = OrchestratorWorkerPattern()
        self.prompt_chaining_agent = PromptChainingPattern()

    # ------------------------------------------------------------------ #
    # Message generation
    # ------------------------------------------------------------------ #
    def generate_swift_messages(self) -> List[Dict]:
        """Generate SWIFT messages and normalise them to plain dicts.

        The generator returns ``SWIFTMessage`` pydantic models, but the agent
        patterns operate on dicts (``message.get(...)``) and serialise them to
        JSON for the LLM. We convert once here so the rest of the pipeline has
        a single, consistent representation.
        """
        messages = self.swift_generator.generate_messages(
            count=self.config.MESSAGE_COUNT,
            bank_count=self.config.BANK_COUNT,
        )
        return [
            m.model_dump(mode="json") if hasattr(m, "model_dump") else dict(m)
            for m in messages
        ]

    # ------------------------------------------------------------------ #
    # Pipeline stages
    # ------------------------------------------------------------------ #
    def process_with_evaluator_optimizer(self, messages: List[Dict]) -> List[Dict]:
        """Step 1: Validate and correct SWIFT messages."""
        print("\n" + "=" * 60)
        print("STEP 1: EVALUATOR-OPTIMIZER PATTERN")
        print("=" * 60)
        return self.evaluator_optimizer.process_with_evaluator_optimizer(messages)

    def process_with_parallelization(self, messages: List[Dict]) -> List[Dict]:
        """Step 2: Concurrent multi-agent fraud detection."""
        print("\n" + "=" * 60)
        print("STEP 2: PARALLELIZATION PATTERN")
        print("=" * 60)
        return self.parallelization_agent.process_batch_parallel(messages)

    def process_with_prompt_chaining(self, messages: List[Dict]) -> Dict:
        """Step 3: Deep multi-stage fraud analysis via prompt chaining.

        Only suspicious transactions are escalated to the (relatively expensive)
        chain, which is both a performance optimization and mirrors how a real
        fraud desk triages work.
        """
        print("\n" + "=" * 60)
        print("STEP 3: PROMPT CHAINING PATTERN")
        print("=" * 60)

        suspicious = [
            m for m in messages
            if m.get('fraud_status') == 'FRAUDULENT'
            or (m.get('fraud_score') or 0) >= 30
        ]
        if not suspicious:
            print("No suspicious transactions to escalate; skipping deep chain.")
            return {}

        print(f"Escalating {len(suspicious)}/{len(messages)} suspicious transaction(s) "
              "to the investigation chain...")
        return self.prompt_chaining_agent.process_chain(suspicious)

    def process_with_orchestrator_worker(self, messages: List[Dict],
                                         filter_name: str = "non_fraudulent") -> Dict:
        """Step 4: Delegate final processing of a filtered message set.

        TODO 5 is fulfilled here: the message subset is chosen by a named
        filter, letting ``run()`` produce two distinct report sets from one
        processed batch.
        """
        print("\n" + "=" * 60)
        print(f"STEP 4: ORCHESTRATOR-WORKER PATTERN  (filter: {filter_name})")
        print("=" * 60)

        predicate = self.FILTERS.get(filter_name, self.FILTERS["non_fraudulent"])
        clean_messages = [msg for msg in messages if predicate(msg)]
        print(f"Selected {len(clean_messages)}/{len(messages)} messages for '{filter_name}'.")

        result = self.orchestrator_worker.process_with_orchestrator(clean_messages)
        return {
            "filter": filter_name,
            "message_count": len(clean_messages),
            "result": result,
        }

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def run(self, report_filters=("non_fraudulent", "high_value")):
        """Main execution: run all patterns in sequence and emit two reports."""
        try:
            print("=" * 60)
            print("SWIFT TRANSACTION PROCESSING SYSTEM")
            print("=" * 60)
            if not Config.has_api_key():
                print("\n[NOTE] No live OpenAI key detected -> running in OFFLINE "
                      "fallback mode (deterministic heuristics).")

            print("\nGenerating SWIFT messages...")
            messages = self.generate_swift_messages()
            print(f"Generated {len(messages)} SWIFT messages")

            # TODO 1: Evaluator-Optimizer validation & correction.
            validated_messages = self.process_with_evaluator_optimizer(messages)

            # TODO 2: Parallel multi-agent fraud detection.
            processed_messages = self.process_with_parallelization(validated_messages)

            # TODO 3: Prompt-chaining deep analysis.
            chain_results = self.process_with_prompt_chaining(processed_messages)

            # TODO 4 + TODO 5: Orchestrator-worker for TWO different report sets.
            orchestrator_sets = []
            for filter_name in report_filters:
                orchestrator_sets.append(
                    self.process_with_orchestrator_worker(processed_messages, filter_name)
                )

            # Final reporting (STP rate + per-transaction detail).
            print("\n" + "=" * 60)
            print("GENERATING REPORTS")
            print("=" * 60)

            # Primary report: the full processed batch.
            report = self.report_generator.generate(
                report_name="pipeline_report",
                messages=processed_messages,
                chain_results=chain_results,
                orchestrator_sets=orchestrator_sets,
            )
            metrics = report["metrics"]
            print(f"STP rate: {metrics['stp_rate_pct']}%  "
                  f"({metrics['stp_count']}/{metrics['total_messages']} straight-through)")
            print(f"Primary report:     {report['_paths']['json']}")
            print(f"                    {report['_paths']['txt']}")

            # Two additional, distinct report SETS — one per requested filter —
            # written as their own files so each view can be inspected on its own.
            print("\nAdditional report sets:")
            for filter_name in report_filters:
                predicate = self.FILTERS.get(filter_name, self.FILTERS["non_fraudulent"])
                subset = [m for m in processed_messages if predicate(m)]
                subset_report = self.report_generator.generate(
                    report_name=f"{filter_name}_report",
                    messages=subset,
                )
                sm = subset_report["metrics"]
                print(f"  [{filter_name}] {sm['total_messages']} msgs, "
                      f"STP {sm['stp_rate_pct']}%  -> {subset_report['_paths']['json']}")

            print("\n" + "=" * 60)
            print("PROCESSING COMPLETE")
            print("=" * 60)
            return report

        except Exception as e:
            print(f"Error in main execution: {e}")
            raise


def _amount_of(message: Dict) -> float:
    """Best-effort numeric amount extraction from a message dict."""
    try:
        return float(''.join(c for c in str(message.get('amount', '0'))
                             if c.isdigit() or c == '.'))
    except ValueError:
        return 0.0


if __name__ == "__main__":
    system = SWIFTProcessingSystem()
    system.run()
