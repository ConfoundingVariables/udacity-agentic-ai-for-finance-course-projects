"""DSPy tool router: few-shot demos from config.toml, a confidence score, and a feedback loop."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

import dspy

from .models import CONFIG

RUNTIME = Path(__file__).resolve().parent.parent / "runtime"
FEEDBACK, SAVED = RUNTIME / "router_feedback.jsonl", RUNTIME / "router.json"


class RouteQuery(dspy.Signature):
    """Select the tool(s) needed to fully answer a financial question."""

    query: str = dspy.InputField()
    tools: list[str] = dspy.OutputField(desc="exact tool names to call")
    confidence: float = dspy.OutputField(desc="0-1: how sure you are these tools fully answer the query")


def _matches(example, pred, trace=None) -> bool:
    return set(pred.tools) == set(example.tools)


class ToolRouter:
    """Routes queries; learns from feedback via optimize()."""

    def __init__(self, catalog: str):
        self.signature = RouteQuery.with_instructions(
            f"{RouteQuery.__doc__}\n\nGuidelines:{CONFIG['routing']['guidelines']}\nTools:\n{catalog}")
        self.program = dspy.ChainOfThought(self.signature)
        if SAVED.exists():
            self.program.load(SAVED)  # learned demos from a previous optimize()
        else:
            self.program = dspy.LabeledFewShot(k=16).compile(self.program, trainset=self._examples())

    def __call__(self, query: str) -> dspy.Prediction:
        return self.program(query=query)

    def _examples(self) -> list[dspy.Example]:
        feedback = [json.loads(line) for line in FEEDBACK.read_text().splitlines()] if FEEDBACK.exists() else []
        return [dspy.Example(query=e["query"], tools=e["tools"], confidence=1.0).with_inputs("query")
                for e in CONFIG["routing"]["examples"] + feedback]

    def record_feedback(self, query: str, correct_tools: list[str]):
        RUNTIME.mkdir(exist_ok=True)
        with FEEDBACK.open("a") as f:
            f.write(json.dumps({"query": query, "tools": correct_tools}) + "\n")

    def _accuracy(self, examples: list[dspy.Example]) -> float:
        with ThreadPoolExecutor(4) as pool:  # Vocareum caps parallel calls per key
            return mean(pool.map(lambda e: _matches(e, self(e.query)), examples))

    def optimize(self) -> dict:
        """Recompile the router with seed examples + feedback as labeled demos, save it, report accuracy."""
        examples = self._examples()
        before = self._accuracy(examples)
        self.program = dspy.LabeledFewShot(k=len(examples)).compile(
            dspy.ChainOfThought(self.signature), trainset=examples)
        self.program.save(SAVED)
        return {"examples": len(examples), "accuracy_before": before, "accuracy_after": self._accuracy(examples)}
