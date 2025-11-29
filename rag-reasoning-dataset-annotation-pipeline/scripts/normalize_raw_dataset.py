# scripts/normalize_raw_dataset.py
from __future__ import annotations
import json, argparse, sys, os
from typing import Any, Dict, List, Set, Tuple
from src.utils import (
    make_record_id, make_doc_id, clean_text,
    merge_snippet, normalize_date, dedup_signature
)

RAW_FILENAME_HINT = "data/raw/conflicts.jsonl"

def normalize_one(raw: Dict[str, Any], out_index: int) -> Dict[str, Any]:
    """
    Convert one DRAG-CONFLICTS item to the target normalized schema.
    out_index is 1-based index in the output (controls '#0001' etc.).
    """
    rec: Dict[str, Any] = {}
    rec["id"] = make_record_id(out_index)

    # query
    query = (raw.get("question") or "").strip()
    if not query:
        raise ValueError("missing question")
    rec["query"] = query

    # conflict_type (copy as-is)
    ctype = (raw.get("conflict_type") or "").strip()
    if not ctype:
        raise ValueError("missing conflict_type")
    rec["conflict_type"] = ctype

    # gold_answer (rename)
    ga = raw.get("correct_answer")
    rec["gold_answer"] = ("" if ga is None else str(ga)).strip()

    # retrieved_docs from search_results
    sr = raw.get("search_results")
    if not isinstance(sr, list) or not sr:
        raise ValueError("empty or invalid search_results")

    docs: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()

    for i, d in enumerate(sr, start=1):
        title      = clean_text(d.get("title"))
        snippet    = clean_text(d.get("snippet"))
        short_text = clean_text(d.get("short_text"))
        merged     = merge_snippet(title, snippet, short_text)

        source_url = (d.get("url") or "").strip()
        ts         = normalize_date(d.get("date"))

        sig = dedup_signature(source_url, merged)
        if sig in seen:
            continue
        seen.add(sig)

        docs.append({
            "doc_id": make_doc_id(i),
            "source_url": source_url,    # verbatim url, can be ""
            "snippet": merged,           # merged Title+Snippet+Short_text (cleaned)
            "timestamp": ts              # "YYYY-MM-DD" | "YYYY" | ""
        })

    if not docs:
        raise ValueError("no valid docs after cleaning/dedup")

    rec["retrieved_docs"] = docs
    return rec

def main():
    ap = argparse.ArgumentParser(description="Normalize DRAG-CONFLICTS raw dataset to the required schema.")
    ap.add_argument("--input", required=True, help=f"path to raw JSONL (e.g., {RAW_FILENAME_HINT})")
    ap.add_argument("--output", required=True, help="path to write normalized JSONL")
    ap.add_argument("--limit", type=int, default=-1, help="process only first N records (debug)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        print(f"Tip: put the official file at: {RAW_FILENAME_HINT}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    in_lines = 0
    out_recs = 0
    tmp_path = args.output + ".tmp"

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(tmp_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            in_lines += 1
            if 0 <= args.limit <= out_recs:
                break

            try:
                raw = json.loads(line)
                norm = normalize_one(raw, out_index=out_recs + 1)
                fout.write(json.dumps(norm, ensure_ascii=False) + "\n")
                out_recs += 1
            except Exception as e:
                # Keep going; warn with index of raw line
                print(f"[WARN] skip raw line {in_lines}: {e}", file=sys.stderr)

    os.replace(tmp_path, args.output)
    print(f"Done. input_lines={in_lines}, output_records={out_recs}")
    print(f"Wrote: {args.output}")

if __name__ == "__main__":
    main()