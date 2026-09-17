"""
Adaptable runtime for the SWIFT processing system.

``process_chat.py`` lets the same agent pipeline run across different scenarios
and inputs without editing code. It supports:

  * Scenario runs          -> full pipeline with a chosen report filter set
  * Ad-hoc single message  -> screen one transaction through the fraud agents
                              (+ deep chain when suspicious)
  * Interactive chat mode  -> a small REPL to drive the above on demand

Examples
--------
    python process_chat.py --scenario default
    python process_chat.py --scenario high_value --count 20
    python process_chat.py --message '{"amount": "9000000 USD", "sender_bic": "TESTUS33XXX", "receiver_bic": "FAKERU22XXX"}'
    python process_chat.py --interactive
"""

import argparse
import json
import uuid
from typing import Dict

from config import Config
from main import SWIFTProcessingSystem
from agents.parallelization import ParallelizationPattern
from agents.prompt_chaining import PromptChainingPattern


# Scenario name -> report filter set used by the full pipeline.
SCENARIOS = {
    "default": ("non_fraudulent", "high_value"),
    "fraud": ("fraudulent", "non_fraudulent"),
    "high_value": ("high_value", "mt103_only"),
    "mt103": ("mt103_only", "non_fraudulent"),
    "mt202": ("mt202_only", "non_fraudulent"),
}


class ProcessChatRuntime:
    """Drives the pipeline across scenarios and ad-hoc inputs."""

    def __init__(self):
        self.parallel = ParallelizationPattern()
        self.chain = PromptChainingPattern()

    # ------------------------------------------------------------------ #
    def run_scenario(self, scenario: str, count: int = None) -> Dict:
        """Run the full pipeline for a named scenario."""
        scenario = scenario if scenario in SCENARIOS else "default"
        if count:
            Config.MESSAGE_COUNT = count
        print(f"\n>>> Running scenario '{scenario}' "
              f"(messages={Config.MESSAGE_COUNT})\n")
        system = SWIFTProcessingSystem()
        return system.run(report_filters=SCENARIOS[scenario])

    # ------------------------------------------------------------------ #
    def screen_message(self, raw: str) -> Dict:
        """Screen a single ad-hoc transaction supplied as JSON or free text."""
        message = self._parse_message(raw)
        print(f"\n>>> Screening message {message['message_id']}\n")

        # Fraud screening via the parallel multi-agent pattern.
        processed = self.parallel.process_batch_parallel([message])
        msg = processed[0]
        print(f"Fraud status : {msg.get('fraud_status')}")
        print(f"Fraud score  : {msg.get('fraud_score')}")
        for reason in msg.get('fraud_reasons', []):
            print(f"  - {reason}")

        # Escalate suspicious messages to the deep investigation chain.
        if msg.get('fraud_status') == 'FRAUDULENT' or (msg.get('fraud_score') or 0) >= 30:
            print("\n>>> Escalating to investigation chain...\n")
            self.chain.process_chain([msg])
            print(f"Chain decision: {msg.get('fraud_decision')} "
                  f"(confidence {msg.get('fraud_confidence')})")
        return msg

    @staticmethod
    def _parse_message(raw: str) -> Dict:
        """Build a message dict from JSON input, filling sensible defaults."""
        defaults = {
            "message_id": f"CHAT-{uuid.uuid4().hex[:8]}",
            "message_type": "MT103",
            "reference": "CHATREF001",
            "amount": "1000.00",
            "currency": "USD",
            "sender_bic": "CHASUS33XXX",
            "receiver_bic": "DEUTDEFFXXX",
            "value_date": "250101",
            "remittance_info": "Ad-hoc screening",
        }
        raw = (raw or "").strip()
        if raw:
            try:
                supplied = json.loads(raw)
                if isinstance(supplied, dict):
                    defaults.update(supplied)
            except json.JSONDecodeError:
                # Treat non-JSON text as remittance info.
                defaults["remittance_info"] = raw
        defaults.setdefault("message_id", f"CHAT-{uuid.uuid4().hex[:8]}")
        return defaults

    # ------------------------------------------------------------------ #
    def interactive(self) -> None:
        """A small REPL supporting multiple runtime scenarios."""
        print("=" * 60)
        print("SWIFT PROCESS-CHAT (interactive)")
        print("Commands:")
        print("  run <scenario>     e.g. 'run high_value'  "
              f"(scenarios: {', '.join(SCENARIOS)})")
        print("  check <json>       screen one transaction")
        print("  help | quit")
        print("=" * 60)

        while True:
            try:
                line = input("swift> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                return
            if not line:
                continue
            cmd, _, rest = line.partition(" ")
            cmd = cmd.lower()
            if cmd in {"quit", "exit", "q"}:
                print("bye")
                return
            if cmd == "help":
                print(f"scenarios: {', '.join(SCENARIOS)}")
            elif cmd == "run":
                self.run_scenario(rest.strip() or "default")
            elif cmd == "check":
                self.screen_message(rest)
            else:
                print(f"Unknown command: {cmd} (try 'help')")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptable SWIFT processing runtime")
    parser.add_argument("--scenario", choices=list(SCENARIOS),
                        help="Run the full pipeline for a named scenario.")
    parser.add_argument("--message", help="Screen a single ad-hoc message (JSON or text).")
    parser.add_argument("--count", type=int, help="Override the number of messages.")
    parser.add_argument("--interactive", action="store_true",
                        help="Start the interactive chat runtime.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runtime = ProcessChatRuntime()

    if args.interactive:
        runtime.interactive()
    elif args.message is not None:
        runtime.screen_message(args.message)
    elif args.scenario:
        runtime.run_scenario(args.scenario, args.count)
    else:
        # Default scenario when invoked with no arguments.
        runtime.run_scenario("default", args.count)


if __name__ == "__main__":
    main()
