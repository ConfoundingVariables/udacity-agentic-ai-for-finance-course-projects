"""
Agent patterns for SWIFT transaction processing
"""

from .evaluator_optimizer import EvaluatorOptimizerPattern
from .orchestrator_worker import OrchestratorWorkerPattern
from .parallelization import ParallelizationPattern
from .prompt_chaining import PromptChainingPattern

__all__ = [
    "EvaluatorOptimizerPattern",
    "OrchestratorWorkerPattern",
    "ParallelizationPattern",
    "PromptChainingPattern"
]
