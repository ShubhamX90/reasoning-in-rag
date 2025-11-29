#!/usr/bin/env python3
"""
eval_text_ccontract_v5.py
-------------------------
Contract-checker for v5 TEXT-MODE generations.

- canon_jsonl: Stage-3 dev schema with retrieved_docs and conflict_type
  (e.g., data/splits/dev.jsonl).
- gens_jsonl: model generations with fields {"id": ..., "raw": ...}
  (from generate_textmode_v5.py, optionally sanitized).

Checks:
- Exactly one <think>...</think> block; no nested tags.
- First top-level JSON array inside <think>:
    • parses as JSON
    • doc_id sequence matches canon retrieved_docs
    • verdict / verdict_reason / source_quality constraints
- A conflict line after that array:
    • <ConflictType> — <reason>
    • reason ≤ max_words_conflict_reason
    • no doc-id ranges like d1–d5
- FINAL section (after </think>):
    • Either abstain (canonical phrase) or normal answer
    • For non-abstain:
        - ≥80% of sentences contain at least one [dX] cite
        - all [dX] within the valid doc range
    • For abstain:
        - must still have a non-empty line
        - abstain violation if any doc verdict is supports/partially supports
- Sentinel [[END-OF-ANSWER]] must appear somewhere in the tail.

Usage:
  python code/eval/eval_text_ccontract_v5.py \
    --canon_jsonl data/splits/dev.jsonl \
    --gens_jsonl  outputs/dev_generations/sft_qlora_v5_run2.sanitized.jsonl \
    --report_json outputs/eval_reports/sft_qlora_v5_run2.text_contract.json
"""

import re
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

THINK_OPEN = re.compile(r"<think>\s*", re.IGNORECASE)
THINK_CLOSE = re.compile(r"\s*</think>", re.IGNORECASE)
DOC_RANGE = re.compile(r"d\d+\s*[–-]\s*d\d+", re.IGNORECASE)
CITE = re.compile(r"\[d(\d+)\]")
ZERO_WIDTH = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")

CONFLICT_TYPES = [
    "No conflict",
    "Complementary information",
    "Conflicting opinions or research outcomes",
    "Conflict due to outdated information",
    "Conflict due to misinformation",
]
TYPE_TO_IDX = {t: i for i, t in enumerate(CONFLICT_TYPES)}
IDX_TO_TYPE = {i: t for t, i in TYPE_TO_IDX.items()}

ABSTAIN_CANON = "CANNOT ANSWER, INSUFFICIENT EVIDENCE"
ABSTAIN_PAT = re.compile(
    r"^\s*cannot\s+answer\s*[,:\-]?\s*insufficient\s+evidence\.?\s*$",
    re.IGNORECASE,
)


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


def extract_think_block(text: str) -> Tuple[Optional[str], Optional[str]]:
    text = ZERO_WIDTH.sub("", text or "")
    m1 = THINK_OPEN.search(text)
    m2 = THINK_CLOSE.search(text)
    if not m1 or not m2 or m2.start() <= m1.end():
        return None, "think_block_missing_or_misaligned"
    before = text[: m1.start()] + text[m2.end() :]
    if THINK_OPEN.search(before) or THINK_CLOSE.search(before):
        return None, "think_block_not_unique"
    return text[m1.end() : m2.start()], None


def json_array_from_block(block: str) -> Tuple[Optional[List[Any]], Optional[str], Optional[int]]:
    start = block.find("[]")
    start = block.find("[") if start == -1 else start
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
    return arr, None, end_idx


def conflict_line_from_block(block: str, max_conflict_reason_words: int) -> Tuple[Optional[Tuple[str, str]], Optional[str]]:
    arr, err, end_idx = json_array_from_block(block)
    if err:
        return None, "no_array_for_conflict_line"
    tail = block[end_idx:].strip()
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    if not lines:
        return None, "conflict_line_missing"
    line = lines[0]
    # Accept em dash, en dash, hyphen, or ':' (but normalize in sanitization)
    if " — " in line:
        t, r = line.split(" — ", 1)
    elif " – " in line:
        t, r = line.split(" – ", 1)
    elif " - " in line:
        t, r = line.split(" - ", 1)
    elif ":" in line:
        t, r = line.split(":", 1)
    else:
        return None, "conflict_line_bad_dash"
    if t not in CONFLICT_TYPES:
        return None, "conflict_type_invalid"
    if max_conflict_reason_words > 0 and len(r.split()) > max_conflict_reason_words:
        return None, "conflict_reason_too_long"
    if DOC_RANGE.search(line):
        return None, "doc_range_in_conflict_reason"
    return (t, r), None


def sentence_split(s: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", s.strip())
    return [p for p in parts if p]


def is_abstain_tail(tail: str) -> bool:
    t = ZERO_WIDTH.sub("", tail or "")
    t = t.replace("[[END-OF-ANSWER]]", "")
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return False
    first = lines[0]
    if first == ABSTAIN_CANON:
        return True
    if ABSTAIN_PAT.match(first):
        return True
    return any(ln == ABSTAIN_CANON for ln in lines)


def validate_example(
    gen: Dict[str, Any],
    canon: Dict[str, Any],
    max_verdict_reason_words: int,
    max_conflict_reason_words: int,
) -> List[str]:
    text = gen.get("raw", "")
    problems: List[str] = []

    # sentinel check
    if "[[END-OF-ANSWER]]" not in text:
        problems.append("sentinel_missing")

    block, err = extract_think_block(text)
    if err:
        problems.append(err)
        return problems

    arr, err, _ = json_array_from_block(block)
    if err:
        problems.append(err)
        return problems

    exp_docs = [d["doc_id"] for d in canon.get("retrieved_docs", [])]
    if [o.get("doc_id") for o in arr] != exp_docs:
        problems.append("doc_id_order_or_membership_mismatch")

    allowed_verdicts = {"supports", "partially supports", "irrelevant"}
    allowed_sq = {"high", "low"}
    any_support = False

    for i, o in enumerate(arr, 1):
        if not isinstance(o, dict):
            problems.append(f"array_item_[{i}]_not_object")
            continue
        if DOC_RANGE.search(json.dumps(o, ensure_ascii=False)):
            problems.append("doc_range_in_array")
        if o.get("doc_id") != f"d{i}":
            problems.append(f"doc_id_not_d{i}")
        v = (o.get("verdict", "") or "").strip().lower()
        if v not in allowed_verdicts:
            problems.append(f"bad_verdict_{o.get('doc_id')}")
        if v in {"supports", "partially supports"}:
            any_support = True
        vr = (o.get("verdict_reason", "") or "").strip()
        if max_verdict_reason_words > 0 and len(vr.split()) > max_verdict_reason_words:
            problems.append(f"verdict_reason_too_long_{o.get('doc_id')}")
        sq = (o.get("source_quality", "") or "").strip().lower()
        if sq not in allowed_sq:
            problems.append(f"bad_source_quality_{o.get('doc_id')}")

    _, err = conflict_line_from_block(block, max_conflict_reason_words=max_conflict_reason_words)
    if err:
        problems.append(err)

    end = THINK_CLOSE.search(text)
    tail = text[end.end() :] if end else ""
    abstaining = is_abstain_tail(tail)

    if abstaining and any_support:
        problems.append("abstain_violation_support_present")

    if not abstaining:
        tail_clean = ZERO_WIDTH.sub("", tail.replace("[[END-OF-ANSWER]]", "")).strip()
        lines = [ln for ln in tail_clean.splitlines() if ln.strip() != ""]
        if not lines:
            problems.append("final_answer_missing")
            return problems
        final = " ".join(lines)
        sents = sentence_split(final)
        if sents:
            cited = sum(1 for s in sents if CITE.search(s))
            cov = cited / max(1, len(sents))
            if cov < 0.8:
                problems.append(f"citation_coverage_lt_0.8:{cov:.2f}")
        max_id = len(exp_docs)
        for m in CITE.finditer(final):
            dnum = int(m.group(1))
            if dnum < 1 or dnum > max_id:
                problems.append("citation_out_of_bounds")
                break
    else:
        tail2 = tail.replace("[[END-OF-ANSWER]]", "").strip()
        if not tail2:
            problems.append("final_answer_missing")

    return problems


def macro_f1_from_conf(conf: List[List[int]]) -> Tuple[float, List[float], List[float], List[float]]:
    n = len(conf)
    precs, recs, f1s = [], [], []
    for c in range(n):
        tp = conf[c][c]
        fp = sum(conf[r][c] for r in range(n)) - tp
        fn = sum(conf[c][r] for r in range(n)) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
    macro = sum(f1s) / n if n > 0 else 0.0
    return macro, precs, recs, f1s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon_jsonl", required=True)
    ap.add_argument("--gens_jsonl", required=True)
    ap.add_argument("--report_json", required=True)
    ap.add_argument("--max_words_verdict_reason", type=int, default=80)
    ap.add_argument("--max_words_conflict_reason", type=int, default=50,  # v5: 50 words
                    help="Max words allowed in conflict_reason (v5: 50)")
    args = ap.parse_args()

    canon_by_id = {ex["id"]: ex for ex in read_jsonl(args.canon_jsonl)}
    total = 0
    ok = 0
    abstain = 0
    problems_log = []

    # text-mode label accuracy
    conf = [[0] * len(CONFLICT_TYPES) for _ in range(len(CONFLICT_TYPES))]
    label_pairs = 0

    for rec in read_jsonl(args.gens_jsonl):
        total += 1
        cid = rec.get("id")
        if cid not in canon_by_id:
            problems_log.append({"id": cid, "problems": ["id_missing_in_canon"]})
            continue

        canon = canon_by_id[cid]
        probs = validate_example(
            rec,
            canon,
            max_verdict_reason_words=args.max_words_verdict_reason,
            max_conflict_reason_words=args.max_words_conflict_reason,
        )
        if not probs:
            ok += 1
        else:
            problems_log.append({"id": cid, "problems": probs})

        # abstain stats
        raw = rec.get("raw", "")
        end = THINK_CLOSE.search(raw)
        tail = raw[end.end() :] if end else ""
        if is_abstain_tail(tail):
            abstain += 1

        # text-mode label F1: predicted conflict type vs gold conflict_type (if present)
        block, err = extract_think_block(raw)
        gold = canon.get("conflict_type", None)
        if block and not err and isinstance(gold, str) and gold in TYPE_TO_IDX:
            pred_tuple, _ = conflict_line_from_block(
                block, max_conflict_reason_words=args.max_words_conflict_reason
            )
            if pred_tuple:
                pred = pred_tuple[0]
                if pred in TYPE_TO_IDX:
                    conf[TYPE_TO_IDX[gold]][TYPE_TO_IDX[pred]] += 1
                    label_pairs += 1

    summary: Dict[str, Any] = {
        "total": total,
        "ok_all_checks": ok,
        "ok_rate_pct": round(100 * ok / max(1, total), 1),
        "abstain_count": abstain,
        "problems": problems_log[:50],
    }

    if label_pairs > 0:
        macro, precs, recs, f1s = macro_f1_from_conf(conf)
        summary["label_f1"] = {
            "pairs_evaluated": label_pairs,
            "macro_f1": round(macro, 4),
            "per_class": {
                IDX_TO_TYPE[i]: {
                    "precision": round(precs[i], 4),
                    "recall": round(recs[i], 4),
                    "f1": round(f1s[i], 4),
                }
                for i in range(len(CONFLICT_TYPES))
            },
            "confusion_matrix": {
                IDX_TO_TYPE[i]: {IDX_TO_TYPE[j]: conf[i][j] for j in range(len(CONFLICT_TYPES))}
                for i in range(len(CONFLICT_TYPES))
            },
        }

    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()