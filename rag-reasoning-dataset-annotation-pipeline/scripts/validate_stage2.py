"""
Stage-2 Output Validation & Audit (Enhanced)
============================================
Validates Stage-2 reasoning outputs and produces a detailed report.

Now includes:
  • Schema validation & logical consistency checks
  • Word-limit enforcement for `conflict_reason`
  • Counts for answerable vs. not-answerable cases
  • Detection of all-irrelevant-document cases
  • Identification and pretty-printed listing of
    all not-answerable records (with full detail)
  • Optional JSON report output for reproducibility

Usage:
  python -m scripts.validate_stage2 \
    --input ~/Desktop/rag-reasoning-dataset/data/stage2_outputs/stage2_final.jsonl \
    --report ~/Desktop/rag-reasoning-dataset/data/stage2_outputs/stage2_validation_report.json
"""

import json
import re
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List

# -------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------

def word_count(s: str) -> int:
    """Count word-like tokens in a string."""
    return len(re.findall(r"\b\w+\b", s or ""))

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL safely, ignoring malformed lines."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[!] Line {i} is not valid JSON: {e}")
    return rows

def all_irrelevant(per_doc_notes: List[Dict[str, Any]]) -> bool:
    """Return True if all per-doc verdicts == 'irrelevant'."""
    if not per_doc_notes:
        return False
    return all(str(d.get("verdict", "")).strip().lower() == "irrelevant" for d in per_doc_notes)

# -------------------------------------------------------------
# Validation rules
# -------------------------------------------------------------

def validate_record(rec: Dict[str, Any]) -> List[str]:
    """
    Validate one Stage-2 record and return list of issues found.
    Covers schema completeness, type checks, word limits, and
    logical consistency of evidence vs. answerability.
    """
    errs = []

    required = ["id", "query", "conflict_type", "per_doc_notes",
                "conflict_reason", "answerable_under_evidence"]
    for field in required:
        if field not in rec:
            errs.append(f"Missing field: {field}")

    # --- conflict_reason
    reason = rec.get("conflict_reason", "")
    if not isinstance(reason, str):
        errs.append("conflict_reason must be a string")
    elif not reason.strip():
        errs.append("conflict_reason is empty")
    elif word_count(reason) > 100:
        errs.append(f"conflict_reason exceeds 100 words ({word_count(reason)})")

    # --- answerable flag
    ans_flag = rec.get("answerable_under_evidence")
    if not isinstance(ans_flag, bool):
        errs.append("answerable_under_evidence must be boolean")

    verdicts = [d.get("verdict", "").lower() for d in rec.get("per_doc_notes", []) if isinstance(d, dict)]
    has_support = any(v in {"supports", "partially supports"} for v in verdicts)
    all_irrel = all(v == "irrelevant" for v in verdicts) if verdicts else False

    if ans_flag and not has_support:
        errs.append("answerable_under_evidence=True but no supporting or partially supporting doc")
    if not ans_flag and has_support:
        errs.append("answerable_under_evidence=False but supporting evidence exists")
    if all_irrel and ans_flag:
        errs.append("answerable_under_evidence=True but all docs are irrelevant")

    return errs

# -------------------------------------------------------------
# Main validation routine
# -------------------------------------------------------------

def run_validation(input_path: str, report_path: str):
    path = Path(input_path)
    assert path.exists(), f"File not found: {path}"

    records = load_jsonl(path)
    print(f"\n🔍 Loaded {len(records)} Stage-2 records from {path}")

    total_errs = 0
    issue_map = {}
    answerable_counts = Counter()
    conflict_type_counts = Counter()
    all_irrelevant_cases = []
    not_answerable_records = []  # store full detail of not-answerable cases

    for rec in records:
        errs = validate_record(rec)
        if errs:
            issue_map[rec.get("id", "???")] = errs
            total_errs += len(errs)

        ans_flag = rec.get("answerable_under_evidence", None)
        if isinstance(ans_flag, bool):
            if ans_flag:
                answerable_counts["answerable"] += 1
            else:
                answerable_counts["not_answerable"] += 1
                not_answerable_records.append(rec)

        conflict_type_counts[rec.get("conflict_type", "UNKNOWN")] += 1

        if all_irrelevant(rec.get("per_doc_notes", [])):
            all_irrelevant_cases.append(rec.get("id", "???"))

    # ---------------------------------------------------------
    # Summary printing
    # ---------------------------------------------------------
    print("\n===================== VALIDATION SUMMARY =====================")
    print(f"Total records validated: {len(records)}")
    print(f"Records with any issue  : {len(issue_map)} ({(len(issue_map)/len(records))*100:.1f}%)")
    print(f"Total individual issues : {total_errs}")
    print("--------------------------------------------------------------")
    print("Answerability stats:")
    for k, v in answerable_counts.items():
        print(f"  {k:20s} : {v}")
    print("--------------------------------------------------------------")
    print("Conflict Type distribution:")
    for k, v in conflict_type_counts.items():
        print(f"  {k:35s} : {v}")
    print("--------------------------------------------------------------")
    print(f"Cases where ALL docs are irrelevant: {len(all_irrelevant_cases)}")
    if all_irrelevant_cases:
        print("Example IDs:", ", ".join(all_irrelevant_cases[:10]))
    print("==============================================================\n")

    # ---------------------------------------------------------
    # Detailed not-answerable printout
    # ---------------------------------------------------------
    if not_answerable_records:
        print(f"\n================= NOT ANSWERABLE RECORDS ({len(not_answerable_records)}) =================")
        for rec in not_answerable_records:
            print(json.dumps(rec, indent=2, ensure_ascii=False))
            print("--------------------------------------------------------------------")
    else:
        print("✅ No not-answerable records found.\n")

    # ---------------------------------------------------------
    # JSON report output
    # ---------------------------------------------------------
    report = {
        "summary": {
            "total_records": len(records),
            "records_with_issues": len(issue_map),
            "total_issues": total_errs,
            "answerable_counts": dict(answerable_counts),
            "conflict_type_distribution": dict(conflict_type_counts),
            "all_irrelevant_case_count": len(all_irrelevant_cases),
            "not_answerable_count": len(not_answerable_records),
        },
        "invalid_records": issue_map,
        "all_irrelevant_cases": all_irrelevant_cases,
        "not_answerable_records": not_answerable_records,
    }

    out_path = Path(report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        json.dump(report, fout, ensure_ascii=False, indent=2)

    print(f"✅ Detailed validation report written to: {out_path}\n")

# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Validate Stage-2 reasoning outputs in detail.")
    ap.add_argument("--input", required=True, help="Path to stage2_final.jsonl")
    ap.add_argument("--report", required=True, help="Path to save validation report JSON")
    args = ap.parse_args()

    run_validation(args.input, args.report)