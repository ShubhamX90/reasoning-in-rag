"""
Stage-3 Validation Script
-------------------------
Validates Stage-3 JSONL outputs for structural and logical consistency.
Now includes rule:
→ When abstain = true, answer must be "CANNOT ANSWER, INSUFFICIENT EVIDENCE"
and prints all IDs where abstain occurred.
"""

import json
import argparse
from pathlib import Path

def validate_stage3(path: str):
    ids_with_abstain = []
    total = 0
    valid = 0
    invalid = 0
    issues = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                rec = json.loads(line)
            except Exception as e:
                issues.append((None, f"JSON parse error: {e}"))
                invalid += 1
                continue

            rid = rec.get("id", f"unnamed_{total}")

            exp = rec.get("expected_response", {})
            think = rec.get("think", "")
            abstain = exp.get("abstain", False)

            # --- Rule 1: Abstain handling
            if abstain:
                ids_with_abstain.append(rid)
                ans = exp.get("answer", "")
                if ans.strip() != "CANNOT ANSWER, INSUFFICIENT EVIDENCE":
                    issues.append((rid, "Abstain=True but answer field not set to required message"))
                    invalid += 1
                    continue  # skip other checks
                else:
                    valid += 1
                    continue

            # --- Rule 2: Basic required fields
            required_fields = ["answer", "evidence"]
            if not all(k in exp for k in required_fields):
                issues.append((rid, "Missing fields in expected_response"))
                invalid += 1
                continue

            if not isinstance(exp.get("evidence"), list):
                issues.append((rid, "Evidence field not a list"))
                invalid += 1
                continue

            if not isinstance(exp.get("answer"), str):
                issues.append((rid, "Answer field not string"))
                invalid += 1
                continue

            # --- Rule 3: Trace presence
            if not think or not isinstance(think, str):
                issues.append((rid, "Missing or invalid think trace"))
                invalid += 1
                continue

            valid += 1

    print("======================================")
    print(f"✅ Total records checked : {total}")
    print(f"✅ Valid entries          : {valid}")
    print(f"⚠️  Invalid entries        : {invalid}")
    print("======================================")

    if ids_with_abstain:
        print("\n🔹 Abstain cases detected (IDs):")
        for rid in ids_with_abstain:
            print(f"  - {rid}")
    else:
        print("\n✅ No abstain cases found.")

    if issues:
        print("\n⚠️  Detailed issues:")
        for rid, msg in issues:
            print(f"  [{rid}] {msg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Stage-3 output JSONL file")
    parser.add_argument("--input", required=True, help="Path to Stage-3 JSONL output file")
    args = parser.parse_args()
    validate_stage3(args.input)