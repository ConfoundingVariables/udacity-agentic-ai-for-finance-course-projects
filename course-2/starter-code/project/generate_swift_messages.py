"""
Generate sample SWIFT message data.

Produces a structured ``data/swift_messages.csv`` file (using the project's
``SWIFTGenerator``) for downstream processing, and optionally the raw ``.swift``
text messages under ``swift_messages/`` for inspection.

Usage:
    python generate_swift_messages.py            # 100 messages -> data/swift_messages.csv
    python generate_swift_messages.py --count 250
    python generate_swift_messages.py --raw      # also write raw .swift files
"""

import argparse
import os

import pandas as pd

from services.swift_generator import SWIFTGenerator


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "swift_messages.csv")


def _to_raw_mt(message) -> str:
    """Render a structured message as a raw MT block for the .swift files."""
    if message.message_type == "MT103":
        return (
            f"{{1:F01{message.sender_bic}0000000000}}\n"
            f"{{2:I103{message.receiver_bic}N}}\n"
            f"{{4:\n"
            f":20:{message.reference}\n"
            f":23B:CRED\n"
            f":32A:{message.value_date}{message.currency}{message.amount}\n"
            f":50K:{message.ordering_customer or ''}\n"
            f":59:{message.beneficiary or ''}\n"
            f":70:{message.remittance_info or ''}\n"
            f":71A:OUR\n-}}"
        )
    return (
        f"{{1:F01{message.sender_bic}0000000000}}\n"
        f"{{2:I202{message.receiver_bic}N}}\n"
        f"{{4:\n"
        f":20:{message.reference}\n"
        f":21:{message.reference}\n"
        f":32A:{message.value_date}{message.currency}{message.amount}\n"
        f":52A:{message.sender_bic}\n"
        f":58A:{message.receiver_bic}\n-}}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample SWIFT data")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of messages to generate (default: 100)")
    parser.add_argument("--raw", action="store_true",
                        help="Also write raw .swift text files")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    generator = SWIFTGenerator()
    messages = generator.generate_messages(count=args.count)

    # Structured CSV (primary output).
    rows = [m.model_dump(mode="json") for m in messages]
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"✅ Wrote {len(df)} structured messages to {CSV_PATH}")

    # Optional raw .swift files.
    if args.raw:
        raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swift_messages")
        os.makedirs(raw_dir, exist_ok=True)
        for i, message in enumerate(messages, start=1):
            filename = f"{message.message_type}_{i:03d}.swift"
            with open(os.path.join(raw_dir, filename), "w", encoding="utf-8") as fh:
                fh.write(_to_raw_mt(message))
        print(f"✅ Wrote {len(messages)} raw .swift files to {raw_dir}/")


if __name__ == "__main__":
    main()
