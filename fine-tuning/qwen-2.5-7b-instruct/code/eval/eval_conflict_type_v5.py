#!/usr/bin/env python3
"""
eval_conflict_type_v5.py
------------------------
Compares the gold conflict_type (from Stage-3 schema) with the predicted
conflict_type inside <think>...</think> of v5 text-mode generations.

Typical usage:
  python code/eval/eval_conflict_type_v5.py \
    --canon_jsonl data/splits/dev.jsonl \
    --gens_jsonl  outputs/dev_generations/sft_qlora_v5_run2.sanitized.jsonl \
    --report_json outputs/eval_reports/sft_qlora_v5_run2.conflict_type.json \
    --per_id_csv  outputs/eval_reports/sft_qlora_v5_run2.conflict_type_per_id.csv
"""

import re
import json
import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# ---------------- Canonical labels ----------------
CANON_TYPES = [
    "No conflict",
    "Complementary information",
    "Conflicting opinions or research outcomes",
    "Conflict due to outdated information",
    "Conflict due to misinformation",
]
_CANON_LOWER = {lbl.lower(): lbl for lbl in CANON_TYPES}

# Common aliases/typos we normalize (case-insensitive handled below)
_ALIAS_MAP_LOWER = {
    "conflicting opinions and research outcomes": "Conflicting opinions or research outcomes",
    "complementary info": "Complementary information",
    "no conflict": "No conflict",
    "outdated information": "Conflict due to outdated information",
    "conflict due to outdated info": "Conflict due to outdated information",
    "misinformation": "Conflict due to misinformation",
}

# ---------------- Regexes ----------------
THINK_OPEN = re.compile(r"<think>\s*", re.IGNORECASE)
THINK_CLOSE = re.compile(r"\s*</think>", re.IGNORECASE)
# Split on em dash, en dash, or hyphen, with flexible spacing, once
DASH_SPLIT = re.compile(r"\s*[—–-]\s*", re.UNICODE)


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


# ---------------- Normalization ----------------
def normalize_label(s: Optional[str]) -> str:
    """Return canonical label if possible; else original (trimmed)."""
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", " ", s.strip())
    l = s.lower()
    # fix the 'and' vs 'or' and other aliases
    if l in _ALIAS_MAP_LOWER:
        return _ALIAS_MAP_LOWER[l]
    # exact canonical match (case-insensitive)
    if l in _CANON_LOWER:
        return _CANON_LOWER[l]
    return s  # leave as-is (will be treated as unexpected later)


# ---------------- Parse helpers ----------------
def extract_think_block(text: str) -> Tuple[Optional[str], Optional[str]]:
    m1 = THINK_OPEN.search(text or "")
    m2 = THINK_CLOSE.search(text or "")
    if not m1 or not m2 or m2.start() <= m1.end():
        return None, "think_block_missing_or_misaligned"
    # ensure exactly one pair
    before = (text[: m1.start()] or "") + (text[m2.end() :] or "")
    if THINK_OPEN.search(before) or THINK_CLOSE.search(before):
        return None, "think_block_not_unique"
    return text[m1.end() : m2.start()], None


def json_array_end_index(block: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Return (end_idx, err) where end_idx is the char index right after the
    first top-level JSON array closing bracket in `block`.
    """
    if block is None:
        return None, "no_block"
    start = block.find("[")
    if start < 0:
        return None, "no_json_array"
    depth, in_str, esc = 0, False, False
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
        return None, "json_array_unbalanced"
    return end_idx, None


def parse_conflict_type_from_think(think_block: str) -> Tuple[Optional[str], Optional[str]]:
    """
    From the think block, skip the first JSON array, then read the next
    non-empty line and split by an em/en/hyphen dash. Return (ctype, err).
    """
    end_idx, err = json_array_end_index(think_block)
    if err:
        return None, err
    tail = (think_block[end_idx:] or "").strip()
    if not tail:
        return None, "conflict_line_missing"
    # first non-empty line after the array
    first_line = next((ln.strip() for ln in tail.splitlines() if ln.strip()), "")
    if not first_line:
        return None, "conflict_line_missing"

    # split once on any dash variant
    parts = DASH_SPLIT.split(first_line, maxsplit=1)
    if len(parts) < 2:
        return None, "conflict_line_bad_dash"

    raw_type = parts[0].strip().strip('"\'')

    ctype = normalize_label(raw_type)
    if ctype in CANON_TYPES:
        return ctype, None
    return None, "conflict_type_invalid_or_unexpected"


# ---------------- Metrics ----------------
def _build_confusion(labels: List[str]) -> Dict[str, Dict[str, int]]:
    return {a: {p: 0 for p in labels} for a in labels}


def classification_report(rows: List[Dict[str, str]], labels: List[str]):
    cm = _build_confusion(labels)
    gold_counts = Counter()
    pred_counts = Counter()
    correct = 0
    total = 0

    for r in rows:
        g = r["gold"]
        p = r["pred"]
        if g in labels and p in labels:
            cm[g][p] += 1
            gold_counts[g] += 1
            pred_counts[p] += 1
            total += 1
            if g == p:
                correct += 1

    per_class = {}
    for lbl in labels:
        tp = cm[lbl][lbl]
        fp = sum(cm[a][lbl] for a in labels if a != lbl)
        fn = sum(cm[lbl][p] for p in labels if p != lbl)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[lbl] = {
            "support": gold_counts[lbl],
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
        }

    acc = correct / total if total > 0 else 0.0
    return {
        "accuracy": round(acc * 100, 2),
        "support": total,
        "distribution_actual": dict(gold_counts),
        "distribution_pred": dict(pred_counts),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon_jsonl", required=True,
                    help="Stage-3 file with gold conflict_type (e.g., data/splits/dev.jsonl)")
    ap.add_argument("--gens_jsonl", required=True,
                    help="Sanitized v5 generations JSONL")
    ap.add_argument("--report_json", required=True,
                    help="Where to write the JSON report")
    ap.add_argument("--per_id_csv", default="",
                    help="Optional CSV with id,gold,pred,match,err")
    args = ap.parse_args()

    # gold (full dataset or split)
    gold_by_id: Dict[str, str] = {}
    for ex in read_jsonl(args.canon_jsonl):
        cid = ex.get("id")
        gct = normalize_label(ex.get("conflict_type"))
        if cid and isinstance(gct, str) and gct:
            gold_by_id[cid] = gct

    # predictions (dev outputs)
    rows = []
    errors = []
    covered_ids = set()

    for rec in read_jsonl(args.gens_jsonl):
        cid = rec.get("id")
        raw = rec.get("raw", "")
        covered_ids.add(cid)

        gold = gold_by_id.get(cid, None)
        if gold is None:
            errors.append({"id": cid, "error": "id_missing_in_gold"})
            continue

        think_block, err = extract_think_block(raw)
        if err:
            rows.append(
                {"id": cid, "gold": gold, "pred": "PRED_MISSING", "match": 0, "err": err}
            )
            continue

        pred, perr = parse_conflict_type_from_think(think_block)
        if perr or pred not in CANON_TYPES:
            rows.append(
                {
                    "id": cid,
                    "gold": gold,
                    "pred": "PRED_INVALID",
                    "match": 0,
                    "err": perr or "invalid_pred",
                }
            )
            continue

        match = 1 if pred == gold else 0
        rows.append({"id": cid, "gold": gold, "pred": pred, "match": match, "err": ""})

    # metrics
    report = classification_report(rows, CANON_TYPES)

    # top confusions
    mismatches = [
        r
        for r in rows
        if r["match"] == 0
        and r["pred"] in (CANON_TYPES + ["PRED_INVALID", "PRED_MISSING"])
    ]
    pair_counts = Counter((r["gold"], r["pred"]) for r in mismatches)
    top_pairs = [
        {"gold": g, "pred": p, "count": c}
        for (g, p), c in pair_counts.most_common(10)
    ]

    full = {
        "totals": {
            "evaluated_ids": len(rows),
            "unique_ids_in_outputs": len(covered_ids),
            "errors_count": len(errors),
        },
        "overall": report,
        "top_confusions": top_pairs,
        "mismatches_sample": mismatches[:25],
    }

    # write JSON
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)

    # optional CSV
    if args.per_id_csv:
        Path(args.per_id_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.per_id_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "gold", "pred", "match", "err"])
            w.writeheader()
            for r in rows:
                w.writerow(r)

    print(json.dumps(full, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()