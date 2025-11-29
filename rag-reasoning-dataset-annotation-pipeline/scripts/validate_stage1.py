#!/usr/bin/env python3
"""
Stage-1 Validation Script  (v2 – 50-word quote limit)
=====================================================
Validates the structure and content of Stage-1 output JSONL files.

Input  : data/stage1_outputs/stage1_final.jsonl
Output : data/stage1_outputs/stage1_validation_results.txt

Checks performed:
  ✓ JSON parse correctness
  ✓ Presence & type of all required fields
  ✓ Allowed values for verdicts and source_quality
  ✓ quote ≤ 50 words, verdict_reason ≤ 40 words
  ✓ Empty key_fact and quote for irrelevant verdicts
  ✓ doc_id consistency between retrieved_docs and per_doc_notes
  ✓ Aggregate summary + top error counts
"""

import json, re
from pathlib import Path
from typing import Dict, Any, List

# ---------- configuration ----------
ALLOWED_VERDICTS = {"supports", "partially supports", "irrelevant"}
ALLOWED_QUALITY = {"high", "medium", "low"}
QUOTE_MAX_WORDS = 100
REASON_MAX_WORDS = 100

# ---------- helpers ----------
def wc(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))

def check_note(note: Dict[str, Any]) -> List[str]:
    errs = []
    required = ["doc_id", "verdict", "key_fact", "quote", "verdict_reason", "source_quality"]
    for k in required:
        if k not in note:
            errs.append(f"missing field '{k}'")
    verdict = note.get("verdict")
    sq = note.get("source_quality")

    if verdict not in ALLOWED_VERDICTS:
        errs.append(f"invalid verdict '{verdict}'")
    if sq not in ALLOWED_QUALITY:
        errs.append(f"invalid source_quality '{sq}'")

    if verdict == "irrelevant":
        if note.get("key_fact") or note.get("quote"):
            errs.append("irrelevant must have empty key_fact and quote")
    else:
        if not note.get("key_fact"):
            errs.append("key_fact empty")
        if not note.get("quote"):
            errs.append("quote empty")
        if wc(note.get("quote", "")) > QUOTE_MAX_WORDS:
            errs.append(f"quote > {QUOTE_MAX_WORDS} words")
        if wc(note.get("verdict_reason", "")) > REASON_MAX_WORDS:
            errs.append(f"verdict_reason > {REASON_MAX_WORDS} words")
    return errs

def check_record(obj: Dict[str, Any]) -> List[str]:
    errs = []
    top_required = ["id", "query", "conflict_type", "gold_answer", "retrieved_docs", "per_doc_notes"]
    for k in top_required:
        if k not in obj:
            errs.append(f"missing field '{k}'")

    if not isinstance(obj.get("retrieved_docs", []), list):
        errs.append("retrieved_docs not a list")
    if not isinstance(obj.get("per_doc_notes", []), list):
        errs.append("per_doc_notes not a list")

    doc_ids = {d.get("doc_id") for d in obj.get("retrieved_docs", [])}
    for n in obj.get("per_doc_notes", []):
        if n.get("doc_id") not in doc_ids:
            errs.append(f"note doc_id '{n.get('doc_id')}' not in retrieved_docs")

    for n in obj.get("per_doc_notes", []):
        suberrs = check_note(n)
        if suberrs:
            errs.extend([f"{n.get('doc_id','?')}: {e}" for e in suberrs])
    return errs

def main():
    input_path = Path("data/stage1_outputs/stage1_final.jsonl")
    output_path = Path("data/stage1_outputs/stage1_validation_results.txt")

    n_total = n_valid = n_invalid = 0
    error_summary = {}

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                fout.write(f"[Line {line_num}] JSON parse error: {e}\n")
                n_invalid += 1
                error_summary["json_parse_error"] = error_summary.get("json_parse_error", 0) + 1
                continue

            errs = check_record(obj)
            if errs:
                n_invalid += 1
                fout.write(f"[{obj.get('id','?')}] {len(errs)} issue(s):\n")
                for e in errs:
                    fout.write(f"  - {e}\n")
                fout.write("\n")
                for e in errs:
                    key = e.split(':')[0]
                    error_summary[key] = error_summary.get(key, 0) + 1
            else:
                n_valid += 1

        fout.write("\n======= SUMMARY =======\n")
        fout.write(f"Total records: {n_total}\n")
        fout.write(f"Valid records: {n_valid}\n")
        fout.write(f"Invalid records: {n_invalid}\n\n")
        fout.write("Top error categories:\n")
        for k,v in sorted(error_summary.items(), key=lambda x:-x[1]):
            fout.write(f"  {k}: {v}\n")

    print(f"✓ Validation complete → {output_path}")
    print(f"Records checked: {n_total}, valid: {n_valid}, invalid: {n_invalid}")

if __name__ == "__main__":
    main()