"""
Solution / run evaluator.

This module provides an *intelligent evaluation* layer that inspects the output
of a pipeline run (a report produced by :mod:`services.reporting`) and scores it
against a set of weighted criteria, returning an overall score and letter grade.

It complements the Evaluator-Optimizer *pattern* (``agents/evaluator_optimizer.py``,
which validates individual SWIFT messages) by evaluating the system as a whole.

Usage
-----
    python evaluator.py reports/pipeline_report.json
    # or programmatically:
    from evaluator import SolutionEvaluator
    result = SolutionEvaluator().evaluate(report_dict)
"""

import json
import sys
from typing import Dict, List, Any, Callable


class Criterion:
    """A single weighted evaluation criterion."""

    def __init__(self, key: str, description: str, weight: float,
                 scorer: Callable[[Dict[str, Any]], float]):
        self.key = key
        self.description = description
        self.weight = weight
        self.scorer = scorer

    def score(self, report: Dict[str, Any]) -> float:
        """Return a normalised score in [0, 1] for this criterion."""
        try:
            return max(0.0, min(1.0, float(self.scorer(report))))
        except Exception:
            return 0.0


class SolutionEvaluator:
    """Evaluates a pipeline report and produces a weighted score + grade."""

    def __init__(self):
        self.criteria: List[Criterion] = [
            Criterion(
                "pipeline_completeness",
                "All pipeline stages produced metrics",
                0.20,
                lambda r: 1.0 if r.get("metrics", {}).get("total_messages", 0) > 0 else 0.0,
            ),
            Criterion(
                "validation_quality",
                "Share of messages that passed validation",
                0.20,
                self._validation_ratio,
            ),
            Criterion(
                "fraud_screening_coverage",
                "Every message received a fraud verdict (no PENDING)",
                0.20,
                self._fraud_coverage,
            ),
            Criterion(
                "delegation",
                "Orchestrator delegated tasks to workers",
                0.15,
                self._delegation_score,
            ),
            Criterion(
                "two_report_sets",
                "At least two distinct report sets were generated",
                0.15,
                lambda r: 1.0 if len(r.get("orchestrator_sets", [])) >= 2 else 0.0,
            ),
            Criterion(
                "stp_reporting",
                "STP rate is reported",
                0.10,
                lambda r: 1.0 if "stp_rate_pct" in r.get("metrics", {}) else 0.0,
            ),
        ]

    # --- individual scorers -------------------------------------------- #
    @staticmethod
    def _validation_ratio(report: Dict[str, Any]) -> float:
        m = report.get("metrics", {})
        total = m.get("total_messages", 0)
        return (m.get("valid_messages", 0) / total) if total else 0.0

    @staticmethod
    def _fraud_coverage(report: Dict[str, Any]) -> float:
        txns = report.get("transactions", [])
        if not txns:
            return 0.0
        scored = sum(1 for t in txns if t.get("fraud_status") in {"CLEAN", "FRAUDULENT", "HELD"})
        return scored / len(txns)

    @staticmethod
    def _delegation_score(report: Dict[str, Any]) -> float:
        sets = report.get("orchestrator_sets", [])
        if not sets:
            return 0.0
        delegated = sum(1 for s in sets if (s.get("task_count") or 0) > 0)
        return delegated / len(sets)

    # --- public API ---------------------------------------------------- #
    def evaluate(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Score a report dict. Returns score, grade and per-criterion detail."""
        details = []
        weighted_total = 0.0
        for c in self.criteria:
            s = c.score(report)
            weighted_total += s * c.weight
            details.append({
                "criterion": c.key,
                "description": c.description,
                "weight": c.weight,
                "score": round(s, 3),
                "weighted": round(s * c.weight, 3),
            })

        overall = round(weighted_total * 100, 1)
        return {
            "overall_score": overall,
            "grade": self._grade(overall),
            "criteria": details,
        }

    def evaluate_file(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
        return self.evaluate(report)

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 90:
            return "A"
        if score >= 80:
            return "B"
        if score >= 70:
            return "C"
        if score >= 60:
            return "D"
        return "F"

    @staticmethod
    def print_scorecard(result: Dict[str, Any]) -> None:
        print("=" * 60)
        print("SOLUTION EVALUATION SCORECARD")
        print("=" * 60)
        for d in result["criteria"]:
            bar = "#" * int(round(d["score"] * 20))
            print(f"{d['criterion']:<26} {d['score']*100:5.1f}%  {bar}")
        print("-" * 60)
        print(f"OVERALL SCORE: {result['overall_score']}%  (Grade: {result['grade']})")
        print("=" * 60)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "reports/pipeline_report.json"
    try:
        evaluation = SolutionEvaluator().evaluate_file(path)
        SolutionEvaluator.print_scorecard(evaluation)
    except FileNotFoundError:
        print(f"Report not found: {path}\nRun 'python main.py' first to produce a report.")
