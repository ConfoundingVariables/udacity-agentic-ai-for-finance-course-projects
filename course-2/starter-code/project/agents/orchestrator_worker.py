"""
Orchestrator-Worker Pattern for Task Distribution.

An Orchestrator analyses a batch of clean SWIFT messages and decomposes the
high-level goal ("process these transactions") into concrete sub-tasks. Those
tasks are then delegated to a pool of GenericAgent workers based on each
worker's declared capabilities, and executed concurrently.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any

from config import Config
from services import llm_client


class OrchestratorWorkerPattern:
    """
    Implements the orchestrator-worker pattern for SWIFT message processing.
    The orchestrator analyzes messages and creates tasks for generic workers.
    """

    def __init__(self):
        """Initialize the orchestrator-worker pattern."""
        self.config = Config()
        self.model = Config.OPENAI_MODEL

    # ------------------------------------------------------------------ #
    # TODO 15: Orchestrator + GenericAgent + coordination
    # ------------------------------------------------------------------ #
    class Orchestrator:
        """Analyzes messages and decomposes the goal into worker tasks."""

        def __init__(self):
            self.model = Config.OPENAI_MODEL

        def analyze_and_create_tasks(self, messages: List[Dict]) -> Dict:
            """Analyze messages and create a structured task list.

            Uses the LLM to plan tasks, and falls back to a deterministic
            planner (below) when the LLM is unavailable so the pattern always
            produces work to delegate.
            """
            system_prompt = """You are an Orchestrator for SWIFT transaction processing.
            Analyze the provided messages and create specific tasks for workers.

            Task types you can create:
            - compliance_check: Check for regulatory compliance
            - fraud_analysis: Detailed fraud investigation
            - amount_verification: Verify and analyze amounts
            - pattern_detection: Detect unusual patterns
            - summary_report: Create summary reports

            Return JSON with your analysis and a list of specific tasks."""

            user_prompt = f"""Analyze these SWIFT messages and create processing tasks:

            {json.dumps(messages, indent=2, default=str)}

            Return JSON with structure:
            {{
                "analysis": "Your analysis of the message batch",
                "task_count": number,
                "tasks": [
                    {{
                        "task_id": "unique_id",
                        "type": "task_type",
                        "description": "What needs to be done",
                        "priority": "high|medium|low",
                        "data": "relevant data for the task"
                    }}
                ]
            }}"""

            return llm_client.chat_json(
                system_prompt,
                user_prompt,
                fallback_factory=lambda: self._default_plan(messages),
            )

        @staticmethod
        def _default_plan(messages: List[Dict]) -> Dict:
            """Deterministic task plan used when the LLM is offline."""
            total = len(messages)
            amounts = []
            for m in messages:
                try:
                    amounts.append(
                        float(''.join(c for c in str(m.get('amount', '0'))
                                      if c.isdigit() or c == '.'))
                    )
                except ValueError:
                    pass
            high_value = [m for m in messages if _amount_of(m) > 50000]
            tasks = [
                {
                    "task_id": "task_001",
                    "type": "compliance_check",
                    "description": "Screen sender/receiver BICs and parties for AML/sanctions concerns.",
                    "priority": "high",
                    "data": {"message_ids": [m.get('message_id') for m in messages]},
                },
                {
                    "task_id": "task_002",
                    "type": "amount_verification",
                    "description": "Verify amount formatting and flag high-value transfers.",
                    "priority": "medium",
                    "data": {"high_value_ids": [m.get('message_id') for m in high_value],
                             "total_value": round(sum(amounts), 2)},
                },
                {
                    "task_id": "task_003",
                    "type": "pattern_detection",
                    "description": "Detect structuring or repeated-corridor patterns across the batch.",
                    "priority": "medium",
                    "data": {"batch_size": total},
                },
                {
                    "task_id": "task_004",
                    "type": "summary_report",
                    "description": "Produce an operations summary of the processed batch.",
                    "priority": "low",
                    "data": {"batch_size": total},
                },
            ]
            return {
                "analysis": (
                    f"Batch of {total} clean transactions; "
                    f"{len(high_value)} high-value (> $50k). Planned {len(tasks)} tasks "
                    "(offline deterministic plan)."
                ),
                "task_count": len(tasks),
                "tasks": tasks,
            }

    class GenericAgent:
        """A worker that executes tasks matching its declared capabilities."""

        # Maps task types to the persona/system prompt used to execute them.
        PERSONAS = {
            'compliance_check': "You are a Compliance Specialist. Execute the compliance check as described.",
            'fraud_analysis': "You are a Fraud Analyst. Perform detailed fraud analysis as requested.",
            'amount_verification': "You are a Financial Auditor. Verify and analyze the amounts as specified.",
            'amount_analysis': "You are a Financial Auditor. Verify and analyze the amounts as specified.",
            'pattern_detection': "You are a Pattern Analysis Expert. Detect and report unusual patterns.",
            'summary_report': "You are a Report Generator. Create the requested summary report.",
        }

        def __init__(self, name: str = "worker", capabilities: List[str] = None):
            self.name = name
            # "*" means this worker can handle any task type (generalist).
            self.capabilities = capabilities or ["*"]

        def can_handle(self, task_type: str) -> bool:
            return "*" in self.capabilities or task_type in self.capabilities

        def execute_task(self, task: Dict) -> Dict:
            """Execute a task assigned by the orchestrator."""
            task_type = task.get('type', 'unknown')
            description = task.get('description', '')
            task_data = task.get('data', {})

            system_prompt = self.PERSONAS.get(
                task_type,
                "You are a Generic Processing Agent. Complete the assigned task.",
            )
            user_prompt = f"""Execute this task:
            Type: {task_type}
            Description: {description}
            Data: {json.dumps(task_data, indent=2, default=str)}

            Return your results in JSON format with keys:
            {{ "findings": "...", "status": "completed", "recommendations": ["..."] }}"""

            def _fallback() -> Dict[str, Any]:
                return {
                    "findings": f"[offline] {description}",
                    "status": "completed",
                    "recommendations": ["Manual review recommended (LLM offline)"],
                }

            result = llm_client.chat_json(system_prompt, user_prompt,
                                          fallback_factory=_fallback)

            return {
                "task_id": task.get('task_id'),
                "type": task_type,
                "worker": self.name,
                "status": "completed",
                "results": result,
            }

    # ------------------------------------------------------------------ #

    def _build_worker_pool(self) -> List["OrchestratorWorkerPattern.GenericAgent"]:
        """Create specialized workers with distinct capabilities."""
        return [
            self.GenericAgent("compliance-worker", ["compliance_check"]),
            self.GenericAgent("fraud-worker", ["fraud_analysis", "pattern_detection"]),
            self.GenericAgent("finance-worker", ["amount_verification", "amount_analysis"]),
            self.GenericAgent("reporting-worker", ["summary_report"]),
            self.GenericAgent("generalist-worker", ["*"]),  # catch-all fallback
        ]

    @staticmethod
    def _assign_worker(task: Dict, workers: List["OrchestratorWorkerPattern.GenericAgent"]):
        """Intelligently delegate a task to the first capable worker."""
        task_type = task.get('type', 'unknown')
        for worker in workers:
            if worker.can_handle(task_type) and "*" not in worker.capabilities:
                return worker
        # Fall back to the generalist worker.
        for worker in workers:
            if "*" in worker.capabilities:
                return worker
        return workers[0]

    def process_with_orchestrator(self, messages: List[Dict]) -> Dict:
        """Process messages using the orchestrator-worker pattern."""
        print("=" * 60)
        print("ORCHESTRATOR-WORKER PATTERN PROCESSING")
        print("=" * 60)

        if not messages:
            print("No messages to process.")
            return {"orchestrator_analysis": {}, "task_results": [],
                    "summary": "No messages to process."}

        # Step 1: Orchestrator plans the work.
        orchestrator = self.Orchestrator()
        print("Orchestrator analyzing messages...")
        orchestrator_response = orchestrator.analyze_and_create_tasks(messages)
        print(f"Orchestrator Analysis: {orchestrator_response.get('analysis', 'No analysis')}")
        tasks = orchestrator_response.get('tasks', [])
        print(f"Tasks created: {len(tasks)}")

        # Step 2: Build a pool of capability-scoped workers.
        workers = self._build_worker_pool()

        # Step 3: Delegate + execute tasks concurrently.
        results = []
        assignments = [(task, self._assign_worker(task, workers)) for task in tasks]
        for task, worker in assignments:
            print(f"  Delegating {task.get('task_id')} ({task.get('type')}) "
                  f"-> {worker.name}")

        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            future_to_task = {
                executor.submit(worker.execute_task, task): task
                for task, worker in assignments
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    results.append(future.result(timeout=90))
                    print(f"  ✓ Task {task.get('task_id')} completed")
                except Exception as exc:  # noqa: BLE001
                    print(f"  ✗ Task {task.get('task_id')} failed: {exc}")
                    results.append({
                        "task_id": task.get('task_id'),
                        "type": task.get('type'),
                        "status": "failed",
                        "error": str(exc),
                    })

        # Step 4: Summarize.
        completed = sum(1 for r in results if r.get('status') == 'completed')
        summary = (
            f"Processed {len(tasks)} tasks for {len(messages)} messages "
            f"({completed} completed) across {len(workers)} workers."
        )
        print("\n" + "-" * 60)
        print(summary)

        return {
            'orchestrator_analysis': orchestrator_response,
            'task_results': results,
            'summary': summary,
        }

    def test_orchestrator(self):
        """Test method for the orchestrator-worker pattern."""
        test_messages = [
            {
                'message_id': 'MSG001',
                'message_type': 'MT103',
                'amount': '75000.00 USD',
                'sender_bic': 'CHASUS33XXX',
                'receiver_bic': 'DEUTDEFFXXX',
                'reference': 'TRX20240101001',
                'remittance_info': 'Payment for equipment purchase',
            },
            {
                'message_id': 'MSG002',
                'message_type': 'MT202',
                'amount': '1000000.00 EUR',
                'sender_bic': 'BNPAFRPPXXX',
                'receiver_bic': 'BARCGB22XXX',
                'reference': 'COV20240101002',
                'remittance_info': 'Cover payment',
            },
        ]

        print("Testing Orchestrator-Worker Pattern\n")
        results = self.process_with_orchestrator(test_messages)

        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        if results:
            print(f"Results obtained: {type(results)}")
            if isinstance(results, dict):
                for key, value in results.items():
                    print(f"{key}: {value if not isinstance(value, list) else f'{len(value)} items'}")

        return results


def _amount_of(message: Dict) -> float:
    """Best-effort numeric amount extraction from a message dict."""
    try:
        return float(''.join(c for c in str(message.get('amount', '0'))
                             if c.isdigit() or c == '.'))
    except ValueError:
        return 0.0


class TestHelper:
    """Helper class for testing individual components."""

    @staticmethod
    def test_orchestrator_only():
        """Test just the Orchestrator class."""
        print("Testing Orchestrator in isolation...")
        orch = OrchestratorWorkerPattern.Orchestrator()
        plan = orch.analyze_and_create_tasks([
            {'message_id': 'X1', 'amount': '60000.00 USD',
             'sender_bic': 'CHASUS33XXX', 'receiver_bic': 'DEUTDEFFXXX'}
        ])
        print(json.dumps(plan, indent=2, default=str))

    @staticmethod
    def test_generic_agent_only():
        """Test just the GenericAgent class."""
        print("Testing GenericAgent in isolation...")
        sample_task = {
            'task_id': 'test_001',
            'type': 'compliance_check',
            'description': 'Check if sender BIC is valid',
            'data': {'sender_bic': 'CHASUS33XXX'},
        }
        agent = OrchestratorWorkerPattern.GenericAgent("compliance-worker",
                                                       ["compliance_check"])
        print(json.dumps(agent.execute_task(sample_task), indent=2, default=str))


if __name__ == "__main__":
    pattern = OrchestratorWorkerPattern()
    pattern.test_orchestrator()
