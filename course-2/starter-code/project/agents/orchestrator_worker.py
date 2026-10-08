"""
Orchestrator-worker pattern for model-selected named batch partitions.

The orchestrator chooses one grouping dimension and names the groups. Each
named group is dispatched as one unit to a capability-scoped worker, retaining
the existing concurrent pool and per-group failure handling.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from config import Config
from services import llm_client


class OrchestratorWorkerPattern:
    """Partition messages into named groups and process each group."""

    def __init__(self):
        self.config = Config()
        self.model = Config.OPENAI_MODEL

    class Orchestrator:
        """Create and validate the model-selected named partition."""

        def __init__(self):
            self.model = Config.OPENAI_MODEL

        def create_grouping_plan(self, messages: List[Dict]) -> Dict:
            """Return ``analysis``, ``grouping_dimension`` and named ``groups``.

            ``groups`` is the complete plan contract. Every input message ID
            occurs exactly once, and no fixed action menu is imposed on the
            model. Invalid input IDs are rejected before dispatch rather than
            silently overwritten by the ID-indexed subset lookup.
            """
            input_ids = [message.get("message_id") for message in messages]
            if any(value is None or not str(value).strip() for value in input_ids):
                raise ValueError("Every message must have a non-empty message_id for grouping")
            normalized_ids = [str(value) for value in input_ids]
            if len(normalized_ids) != len(set(normalized_ids)):
                raise ValueError("message_id values must be unique for grouping")

            system_prompt = """You are the SWIFT transaction operations orchestrator.

Choose the most useful grouping_dimension for this batch, then partition the
messages into named operational groups. The dimension and group names must be
chosen from the evidence in the messages; do not use a predefined action menu.
Every input message_id MUST occur exactly once in one group. Groups MUST be
non-empty. A rationale is optional but useful to reviewers.

Return JSON only with this exact shape:
{
  "analysis": "why this partition is appropriate",
  "grouping_dimension": "the chosen dimension",
  "groups": [
    {
      "group_id": "stable unique ID",
      "name": "human-readable model-chosen group name",
      "message_ids": ["input message_id"],
      "rationale": "optional reason for the grouping"
    }
  ]
}"""
            user_prompt = f"""Partition these SWIFT messages into named groups:

{json.dumps(messages, indent=2, default=str)}

Do not omit or duplicate any message_id."""
            raw = llm_client.chat_json(
                system_prompt,
                user_prompt,
                fallback_factory=lambda: self._default_plan(messages),
            )
            return self._normalize_plan(raw, messages)

        @classmethod
        def _normalize_plan(cls, raw: Any, messages: List[Dict]) -> Dict:
            """Accept only a complete valid partition; otherwise use fallback."""
            if not isinstance(raw, dict) or not isinstance(raw.get("groups"), list):
                return cls._default_plan(messages)
            input_ids = [str(message.get("message_id")) for message in messages]
            known_ids = set(input_ids)
            assigned = set()
            used_group_ids = set()
            groups = []

            for index, candidate in enumerate(raw["groups"], start=1):
                if not isinstance(candidate, dict):
                    return cls._default_plan(messages)
                candidate_ids = candidate.get("message_ids")
                if not isinstance(candidate_ids, list) or not candidate_ids:
                    return cls._default_plan(messages)
                message_ids = [str(value) for value in candidate_ids]
                if (
                    any(value not in known_ids for value in message_ids)
                    or len(message_ids) != len(set(message_ids))
                    or assigned.intersection(message_ids)
                ):
                    return cls._default_plan(messages)
                group_id = str(candidate.get("group_id") or "").strip()
                name = str(candidate.get("name") or "").strip()
                if not group_id or not name or group_id in used_group_ids:
                    return cls._default_plan(messages)
                used_group_ids.add(group_id)
                assigned.update(message_ids)
                group = {"group_id": group_id, "name": name, "message_ids": message_ids}
                if candidate.get("rationale"):
                    group["rationale"] = str(candidate["rationale"]).strip()
                groups.append(group)

            if assigned != known_ids or not groups:
                return cls._default_plan(messages)
            dimension = str(raw.get("grouping_dimension") or "").strip()
            if not dimension:
                return cls._default_plan(messages)
            return {
                "analysis": str(raw.get("analysis") or "Model-selected named batch partition."),
                "grouping_dimension": dimension,
                "groups": groups,
            }

        @staticmethod
        def _default_plan(messages: List[Dict]) -> Dict:
            """Offline fallback: partition by currency, preserving all messages."""
            buckets: Dict[str, List[str]] = {}
            for message in messages:
                currency = _currency_of(message)
                buckets.setdefault(currency, []).append(str(message.get("message_id")))
            groups = [
                {
                    "group_id": f"group_{index:03d}",
                    "name": f"{currency} transactions",
                    "message_ids": message_ids,
                    "rationale": "Offline currency partition derived from the message currency field or amount suffix.",
                }
                for index, (currency, message_ids) in enumerate(buckets.items(), start=1)
            ]
            return {
                "analysis": f"Offline currency partitioned {len(messages)} messages into {len(groups)} named groups.",
                "grouping_dimension": "currency",
                "groups": groups,
            }

    class GenericAgent:
        """Worker that processes one complete named group."""

        PERSONAS = {
            "compliance-worker": "You are a Compliance Specialist.",
            "fraud-worker": "You are a Fraud Analyst.",
            "finance-worker": "You are a Financial Auditor.",
            "reporting-worker": "You are an Operations Reporting Specialist.",
            "generalist-worker": "You are a Generic Processing Agent.",
        }

        def __init__(self, name: str = "worker", capabilities: List[str] = None):
            self.name = name
            self.capabilities = capabilities or ["*"]

        def can_handle(self, capability: str) -> bool:
            return "*" in self.capabilities or capability in self.capabilities

        def process_group(self, task: Dict) -> Dict:
            """Process a group's actual subset with an offline-safe report."""
            group_name = str(task.get("name") or task.get("group_name") or "transaction group")
            messages = list(task.get("messages", []))
            system_prompt = self.PERSONAS.get(self.name, self.PERSONAS["generalist-worker"])
            user_prompt = f"""Process this named SWIFT transaction group.
Grouping dimension: {task.get('grouping_dimension', 'model_defined')}
Group name: {group_name}
Messages:
{json.dumps(messages, indent=2, default=str)}

Return JSON with keys findings, status, recommendations."""

            result = llm_client.chat_json(
                system_prompt,
                user_prompt,
                fallback_factory=lambda: self._offline_group_report(messages, group_name),
            )
            return {
                "group_id": task.get("group_id"),
                "group_name": group_name,
                "grouping_dimension": task.get("grouping_dimension"),
                "message_ids": list(task.get("message_ids", [])),
                "messages": messages,
                "worker": self.name,
                "status": "completed",
                "results": result,
            }

        @staticmethod
        def _offline_group_report(messages: List[Dict], group_name: str) -> Dict[str, Any]:
            totals: Dict[str, float] = {}
            type_distribution: Dict[str, int] = {}
            high_value_ids = []
            fraud_flags = []
            for message in messages:
                currency = _currency_of(message)
                totals[currency] = round(totals.get(currency, 0.0) + _amount_of(message), 2)
                message_type = str(message.get("message_type") or "UNKNOWN")
                type_distribution[message_type] = type_distribution.get(message_type, 0) + 1
                if _amount_of(message) > 50000:
                    high_value_ids.append(message.get("message_id"))
                status = message.get("fraud_status")
                if status and status not in {"CLEAN", "PENDING"}:
                    fraud_flags.append({
                        "message_id": message.get("message_id"),
                        "status": status,
                        "score": message.get("fraud_score"),
                        "reasons": message.get("fraud_reasons", []),
                    })
            return {
                "findings": f"[offline] Processed {len(messages)} messages in {group_name}.",
                "status": "completed",
                "recommendations": ["Manual review recommended (LLM offline)"],
                "amount_totals_by_currency": totals,
                "message_type_distribution": type_distribution,
                "high_value_ids": high_value_ids,
                "fraud_flags": fraud_flags,
            }

    def _build_worker_pool(self) -> List["OrchestratorWorkerPattern.GenericAgent"]:
        """Create the capability-scoped worker pool."""
        return [
            self.GenericAgent("compliance-worker", ["message_type", "compliance"]),
            self.GenericAgent("fraud-worker", ["risk_profile", "fraud"]),
            self.GenericAgent("finance-worker", ["currency", "amount", "amount_band"]),
            self.GenericAgent("reporting-worker", ["reporting", "review_priority"]),
            self.GenericAgent("generalist-worker", ["*"]),
        ]

    @staticmethod
    def _assign_worker(group: Dict, workers: List["OrchestratorWorkerPattern.GenericAgent"]):
        """Route each named group using its model-selected dimension."""
        dimension = str(group.get("grouping_dimension") or "model_defined")
        for worker in workers:
            if worker.can_handle(dimension) and "*" not in worker.capabilities:
                return worker
        for worker in workers:
            if "*" in worker.capabilities:
                return worker
        return workers[0]

    def process_with_orchestrator(self, messages: List[Dict]) -> Dict:
        """Dispatch each named group concurrently and preserve group evidence."""
        print("=" * 60)
        print("ORCHESTRATOR-WORKER PATTERN PROCESSING")
        print("=" * 60)
        if not messages:
            return {
                "orchestrator_analysis": {
                    "analysis": "No messages to process.",
                    "grouping_dimension": "currency",
                    "groups": [],
                },
                "group_results": [],
                "summary": "No messages to process.",
            }

        plan = self.Orchestrator().create_grouping_plan(messages)
        dimension = plan["grouping_dimension"]
        message_by_id = {str(message["message_id"]): message for message in messages}
        workers = self._build_worker_pool()
        assignments = []
        for group in plan["groups"]:
            subset = [message_by_id[mid] for mid in group["message_ids"]]
            group_task = {
                "group_id": group["group_id"],
                "group_name": group["name"],
                "name": group["name"],
                "grouping_dimension": dimension,
                "message_ids": list(group["message_ids"]),
                "messages": subset,
            }
            worker = self._assign_worker(group_task, workers)
            assignments.append((group_task, worker))
            print(f"  Delegating {group_task['group_id']} ({group_task['group_name']}) -> {worker.name}")

        raw_results = {}
        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            future_to_group = {
                executor.submit(worker.process_group, task): task
                for task, worker in assignments
            }
            for future in as_completed(future_to_group):
                task = future_to_group[future]
                group_id = task["group_id"]
                try:
                    raw_results[group_id] = future.result(timeout=90)
                    print(f"  ✓ Group {group_id} completed")
                except Exception as exc:  # noqa: BLE001
                    print(f"  ✗ Group {group_id} failed: {exc}")
                    raw_results[group_id] = {
                        "group_id": group_id,
                        "group_name": task["group_name"],
                        "grouping_dimension": dimension,
                        "message_ids": task["message_ids"],
                        "messages": task["messages"],
                        "worker": next(worker.name for candidate, worker in assignments if candidate is task),
                        "status": "failed",
                        "error": str(exc),
                    }

        group_results = [raw_results[task["group_id"]] for task, _ in assignments]
        completed = sum(1 for result in group_results if result.get("status") == "completed")
        summary = (
            f"Processed {len(group_results)} named groups for {len(messages)} messages "
            f"({completed} completed) using {len(workers)} workers."
        )
        return {
            "orchestrator_analysis": plan,
            "group_results": group_results,
            "summary": summary,
        }

    def test_orchestrator(self):
        """Run a small offline-safe demonstration."""
        sample = [
            {"message_id": "TEST001", "amount": "75000.00 USD"},
            {"message_id": "TEST002", "amount": "500.00 EUR"},
        ]
        result = self.process_with_orchestrator(sample)
        print(json.dumps(result, indent=2, default=str))
        return result


def _currency_of(message: Dict) -> str:
    """Get currency from a field, then from a trailing amount suffix."""
    currency = str(message.get("currency") or "").strip().upper()
    if currency:
        return currency
    amount = str(message.get("amount") or "")
    match = re.search(r"([A-Za-z]{3})\s*$", amount)
    return match.group(1).upper() if match else "UNKNOWN"


def _amount_of(message: Dict) -> float:
    """Best-effort numeric amount extraction from a message dict."""
    try:
        value = "".join(c for c in str(message.get("amount", "0")) if c.isdigit() or c == ".")
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    OrchestratorWorkerPattern().test_orchestrator()
