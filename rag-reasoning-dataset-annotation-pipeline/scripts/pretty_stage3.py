import json
import argparse
from pathlib import Path
from pprint import pprint
import re

def pretty_stage3_entry(file_path: str, query_id: str):
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    found = False
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("id") == query_id:
                found = True
                print("=" * 90)
                print(f"QUERY ID: {record['id']}")
                print("=" * 90)
                print("\nFULL ENTRY:\n")
                pprint(record, sort_dicts=False, width=110)

                think_raw = record.get("think", "")
                clean_think = think_raw.replace("\\n", "\n").strip()

                # Format: add line after each doc reasoning block (JSON object pattern)
                formatted_think = re.sub(r"}\s*,\s*{", "},\n\n{", clean_think)

                # Print reasoning trace with requested spacing
                print("\n" * 5)
                print("REASONING TRACE:\n")
                print("<think>")
                print(formatted_think.replace("<think>", "").replace("</think>", "").strip())
                print("</think>")

                expected = record.get("expected_response", {})
                answer = expected.get("answer", "")
                answer_clean = answer.replace("\\n", "\n").strip()

                print("\nFINAL ANSWER:")
                print(answer_clean)
                print("\n")
                print("=" * 90)
                print("\n")
                break 

    if not found:
        print(f"No record found for ID {query_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretty print one Stage-3 entry by ID")
    parser.add_argument("--id", required=True, help="Query ID (e.g., #0186)")
    parser.add_argument(
        "--input",
        default="data/stage3_outputs/stage3_final.jsonl",
        help="Path to Stage-3 JSONL file (default: data/stage3_outputs/stage3_final.jsonl)"
    )
    args = parser.parse_args()

    pretty_stage3_entry(args.input, args.id)