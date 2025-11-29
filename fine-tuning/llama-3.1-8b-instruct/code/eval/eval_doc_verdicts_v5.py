#!/usr/bin/env python3
"""
Doc-Verdict Evaluation (v5)
---------------------------
Compares per-doc verdicts in model text-mode outputs against the canonical
per_doc_notes verdicts from the Stage-3 dataset.

Usage example:

  python code/eval/eval_doc_verdicts_v5.py \
    --canon_jsonl data/raw/stage3_final.jsonl \
    --gens_jsonl  outputs/dev_generations/sft_qlora_v5.sanitized.jsonl \
    --report_json outputs/eval_reports/doc_verdicts_v5.json \
    --per_doc_csv outputs/eval_reports/doc_verdicts_v5_per_doc.csv

Inputs:
- canon_jsonl: Stage-3 style JSONL with fields:
    {
      "id": "#0134",
      "per_doc_notes": [
        {"doc_id": "d1", "verdict": "supports", ...},
        {"doc_id": "d2", "verdict": "irrelevant", ...},
        ...
      ],
      ...
    }
- gens_jsonl: text-mode generations JSONL (raw or sanitized) with:
    {
      "id": "#0134",
      "raw": "<think>[ {\"doc_id\":\"d1\", \"verdict\":\"supports\", ...}, ... ] ...</think> ... [[END-OF-ANSWER]]"
    }

Output:
- report_json: summary with micro accuracy, confusion matrix, per-class metrics, and some error stats.
- per_doc_csv (optional): one row per evaluated doc (id, doc_id, gold, pred, match, err).

This script does *not* enforce full contract; it only needs the <think> block's
first top-level JSON array.
"""

import re
import json
import argparse
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter

# ---------------- Constants ----------------

ZERO_WIDTH = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")

VERDICTS = ["supports", "partially supports", "irrelevant"]
VERDICT_TO_IDX = {v: i for i, v in enumerate(VERDICTS)}
IDX_TO_VERDICT = {i: v for v, i in VERDICT_TO_IDX.items()}

THINK_OPEN  = re.compile(r"<think>\s*", re.IGNORECASE)
THINK_CLOSE = re.compile(r"\s*</think>", re.IGNORECASE)


# ---------------- IO helpers ----------------

def read_jsonl(p: str):
    with open(p, "r", encoding="utf-8") as f:
        for ln, s in enumerate(f, 1):
            s = s.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                raise ValueError(f"{p}:{ln} bad json: {e}")


# ---------------- Parsing helpers ----------------

def extract_think_block(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (block, err). If ok, block is the text between the single
    <think>...</think>. If not, block=None and err is a reason.
    """
    if text is None:
        return None, "raw_missing"

    text = ZERO_WIDTH.sub("", text)
    m1 = THINK_OPEN.search(text)
    m2 = THINK_CLOSE.search(text)
    if not m1 or not m2 or m2.start() <= m1.end():
        return None, "think_block_missing_or_misaligned"

    before = (text[:m1.start()] or "") + (text[m2.end():] or "")
    if THINK_OPEN.search(before) or THINK_CLOSE.search(before):
        return None, "think_block_not_unique"

    return text[m1.end():m2.start()], None


def json_array_from_block(block: str) -> Tuple[Optional[List[Any]], Optional[str], Optional[int]]:
    """
    Find the first *top-level* JSON array in `block` and parse it.

    Returns (arr, err, end_idx) where:
    - arr: parsed list of objects (or None on error)
    - err: error string or None
    - end_idx: index in block right after the closing ']' of the array
    """
    if block is None:
        return None, "no_block", None

    start = block.find("[")
    if start < 0:
        return None, "no_json_array", None

    depth = 0
    in_str = False
    esc = False
    end_idx = None
    for i in range(start, len(block)):
        ch = block[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break

    if end_idx is None:
        return None, "json_array_unbalanced", None

    arr_text = block[start:end_idx]
    try:
        arr = json.loads(arr_text)
    except Exception as e:
        return None, f"json_array_parse_error: {e}", None

    if not isinstance(arr, list):
        return None, "json_array_not_list", None

    return arr, None, end_idx


def normalize_verdict(v: Any) -> str:
    """
    Normalize verdict strings to one of:
      - "supports"
      - "partially supports"
      - "irrelevant"
    or return "" if unrecognized.
    """
    if not isinstance(v, str):
        return ""
    raw = v.strip().lower()
    # Mild alias handling
    if raw in {"support", "supported"}:
        raw = "supports"
    if raw in {"partial", "partially_supports", "partially-supports"}:
        raw = "partially supports"
    if raw in VERDICT_TO_IDX:
        return raw
    return ""


# ---------------- Metrics helpers ----------------

def macro_f1_from_conf(conf: List[List[int]]) -> Tuple[float, List[float], List[float], List[float]]:
    """
    Compute macro-F1, plus per-class precision/recall/F1.
    conf is a square confusion matrix [gold_idx][pred_idx].
    """
    n = len(conf)
    precs, recs, f1s = [], [], []
    for c in range(n):
        tp = conf[c][c]
        fp = sum(conf[r][c] for r in range(n)) - tp
        fn = sum(conf[c][r] for r in range(n)) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
    macro = sum(f1s) / n if n > 0 else 0.0
    return macro, precs, recs, f1s


# ---------------- Main evaluation ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon_jsonl", required=True,
                    help="Stage-3 style JSONL with id + per_doc_notes")
    ap.add_argument("--gens_jsonl", required=True,
                    help="Text-mode v5 generations (raw or sanitized) JSONL")
    ap.add_argument("--report_json", required=True,
                    help="Where to write the summary JSON")
    ap.add_argument("--per_doc_csv", default="",
                    help="Optional CSV with per-doc rows (id,doc_id,gold,pred,match,err)")
    args = ap.parse_args()

    # -------- Load canon verdicts --------
    canon_by_id: Dict[str, Dict[str, str]] = {}
    for ex in read_jsonl(args.canon_jsonl):
        cid = ex.get("id")
        notes = ex.get("per_doc_notes") or []
        if not isinstance(cid, str):
            continue
        doc_map: Dict[str, str] = {}
        for n in notes:
            if not isinstance(n, dict):
                continue
            did = n.get("doc_id")
            v   = normalize_verdict(n.get("verdict"))
            if isinstance(did, str) and v:
                doc_map[did] = v
        if doc_map:
            canon_by_id[cid] = doc_map

    # -------- Iterate generations --------
    total_examples = 0
    evaluated_examples = 0
    total_doc_pairs = 0
    correct_doc_pairs = 0

    # confusion[gold_idx][pred_idx]
    conf = [[0 for _ in VERDICTS] for _ in VERDICTS]
    error_counts = Counter()
    per_doc_rows = []

    for rec in read_jsonl(args.gens_jsonl):
        total_examples += 1
        cid = rec.get("id")
        raw = rec.get("raw", "")

        if cid not in canon_by_id:
            error_counts["id_missing_in_canon"] += 1
            continue

        gold_docs = canon_by_id[cid]  # dict doc_id -> verdict

        think_block, err = extract_think_block(raw)
        if err:
            error_counts[err] += 1
            continue

        arr, aerr, _ = json_array_from_block(think_block)
        if aerr:
            error_counts[aerr] += 1
            continue

        # Build predicted doc_id -> verdict map
        pred_docs: Dict[str, str] = {}
        for obj in arr:
            if not isinstance(obj, dict):
                error_counts["array_item_not_object"] += 1
                continue
            did = obj.get("doc_id")
            v   = normalize_verdict(obj.get("verdict"))
            if not isinstance(did, str):
                error_counts["doc_id_missing_or_nonstring"] += 1
                continue
            pred_docs[did] = v

        # Evaluate only on intersection of doc_ids that have both gold & pred
        common_doc_ids = sorted(set(gold_docs.keys()) & set(pred_docs.keys()),
                                key=lambda x: (len(x), x))
        if not common_doc_ids:
            error_counts["no_common_doc_ids"] += 1
            continue

        evaluated_examples += 1

        for did in common_doc_ids:
            gold_v = gold_docs.get(did, "")
            pred_v = pred_docs.get(did, "")

            row_err = ""
            if gold_v not in VERDICT_TO_IDX:
                row_err = "gold_verdict_invalid"
                error_counts[row_err] += 1
            elif pred_v not in VERDICT_TO_IDX:
                row_err = "pred_verdict_invalid"
                error_counts[row_err] += 1
            else:
                total_doc_pairs += 1
                g_idx = VERDICT_TO_IDX[gold_v]
                p_idx = VERDICT_TO_IDX[pred_v]
                conf[g_idx][p_idx] += 1
                match = int(gold_v == pred_v)
                if match:
                    correct_doc_pairs += 1
            # For CSV: we still record the row even if invalid
            per_doc_rows.append({
                "id": cid,
                "doc_id": did,
                "gold": gold_v,
                "pred": pred_v,
                "match": int(gold_v == pred_v and row_err == ""),
                "err": row_err,
            })

        # Count missing docs on each side (diagnostic only)
        gold_only = set(gold_docs.keys()) - set(pred_docs.keys())
        pred_only = set(pred_docs.keys()) - set(gold_docs.keys())
        if gold_only:
            error_counts["docs_missing_in_pred"] += len(gold_only)
        if pred_only:
            error_counts["extra_docs_in_pred"] += len(pred_only)

    # -------- Aggregate metrics --------

    if total_doc_pairs > 0:
        micro_acc = correct_doc_pairs / total_doc_pairs
    else:
        micro_acc = 0.0

    macro, precs, recs, f1s = macro_f1_from_conf(conf) if total_doc_pairs > 0 else (0.0, [], [], [])

    metrics = {
        "totals": {
            "total_examples_in_gens": total_examples,
            "examples_with_any_eval": evaluated_examples,
            "total_doc_pairs_evaluated": total_doc_pairs,
            "correct_doc_pairs": correct_doc_pairs,
            "micro_accuracy_doc_level": round(micro_acc * 100, 2),
        },
        "overall": {
            "macro_f1": round(macro, 4),
            "verdict_labels": VERDICTS,
            "per_class": {
                IDX_TO_VERDICT[i]: {
                    "precision": round(precs[i], 4) if precs else 0.0,
                    "recall":    round(recs[i], 4) if recs else 0.0,
                    "f1":        round(f1s[i], 4) if f1s else 0.0,
                }
                for i in range(len(VERDICTS))
            },
            "confusion_matrix": {
                IDX_TO_VERDICT[i]: {
                    IDX_TO_VERDICT[j]: conf[i][j] for j in range(len(VERDICTS))
                }
                for i in range(len(VERDICTS))
            },
        },
        "error_counts": dict(error_counts),
        "notes": {
            "canon_source": args.canon_jsonl,
            "gens_source": args.gens_jsonl,
            "description": "Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.",
        },
    }

    # -------- Write JSON report --------

    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # -------- Optional per-doc CSV --------

    if args.per_doc_csv:
        Path(args.per_doc_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.per_doc_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "doc_id", "gold", "pred", "match", "err"])
            w.writeheader()
            for row in per_doc_rows:
                w.writerow(row)


if __name__ == "__main__":
    main()