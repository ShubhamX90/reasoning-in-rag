#!/usr/bin/env python3
"""
sanitize_text_outputs_v5.py
---------------------------
Post-processes raw text-mode generations to:
- Normalize the per-doc JSON array inside <think>...</think>.
- Enforce:
    • doc_id sequence d1..dN
    • verdict ∈ {supports, partially supports, irrelevant}
    • key_fact == "" when verdict == "irrelevant"
    • verdict_reason ≤ 80 words
    • source_quality ∈ {high, low} (default -> low)
- Normalize the conflict line:
    • <ConflictType> — <reason>
    • reason ≤ 50 words
    • removes doc-id ranges like d1–d5 inside the conflict_reason.
- Drop out-of-bounds citations [dK] in the final answer where K ∉ [1..N].

Usage:
  python code/eval/sanitize_text_outputs_v5.py \
    --in_jsonl  outputs/dev_generations/sft_qlora_v5_run2.raw.jsonl \
    --out_jsonl outputs/dev_generations/sft_qlora_v5_run2.sanitized.jsonl
"""

import re
import json
import argparse
from pathlib import Path
from typing import Tuple, Optional, List, Any

THINK_OPEN = re.compile(r"<think>\s*", re.IGNORECASE)
THINK_CLOSE = re.compile(r"\s*</think>", re.IGNORECASE)
CITE = re.compile(r"\[d(\d+)\]")
ZERO_WIDTH = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")

CONFLICT_TYPES = [
    "No conflict",
    "Complementary information",
    "Conflicting opinions or research outcomes",
    "Conflict due to outdated information",
    "Conflict due to misinformation",
]


def _loose_type(t: str) -> str:
    return r"".join([re.escape(ch) if ch != " " else r"\s+" for ch in t])


CONFLICT_TYPES_RE = "(" + "|".join(_loose_type(t) for t in CONFLICT_TYPES) + ")"
SEP_RE = r"(?:\s*[-–—]{1,2}\s*|\s*:\s*)"
DOC_RANGE = re.compile(r"d\d+\s*[–-]\s*d\d+", re.IGNORECASE)
PUNCT_ONLY = re.compile(r"^[\s,;]+$")


def read_jsonl(p: str):
    with open(p, "r", encoding="utf-8") as f:
        for ln, s in enumerate(f, 1):
            s = s.rstrip("\n")
            if not s.strip():
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                raise ValueError(f"{p}:{ln} bad json: {e}")


def write_jsonl(p: str, items):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def extract_think_span(text: str):
    text = ZERO_WIDTH.sub("", text or "")
    m1 = THINK_OPEN.search(text)
    m2 = THINK_CLOSE.search(text)
    if not m1 or not m2 or m2.start() <= m1.end():
        return None, None, None
    return m1.end(), m2.start(), text[m1.end():m2.start()]


def find_first_top_level_array(block: str):
    start = block.find("[")
    if start < 0:
        return None, None
    depth = 0
    in_str = False
    esc = False
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
                    return start, i + 1
    return None, None


def _normalize_conflict_type(raw: str) -> Optional[str]:
    for t in CONFLICT_TYPES:
        if re.fullmatch(_loose_type(t), raw, flags=re.IGNORECASE):
            return t
    return None


def _trim_words(s: str, n: int = 80) -> str:
    w = s.strip().split()
    return s.strip() if len(w) <= n else " ".join(w[:n])


def _repair_doc_items(arr: list) -> tuple[list, bool]:
    changed = False
    verdict_map = {"support": "supports", "supported": "supports", "partial": "partially supports"}
    for i, obj in enumerate(arr, 1):
        if not isinstance(obj, dict):
            continue
        exp_id = f"d{i}"
        if obj.get("doc_id") != exp_id:
            obj["doc_id"] = exp_id
            changed = True

        v = (obj.get("verdict", "") or "").strip().lower()
        if v in verdict_map:
            obj["verdict"] = verdict_map[v]
            changed = True

        if v == "irrelevant":
            if obj.get("key_fact", None) not in ("", None):
                obj["key_fact"] = ""
                changed = True

        # trim verdict_reason to 80 words
        vr = obj.get("verdict_reason", "")
        if isinstance(vr, str):
            new_vr = _trim_words(vr, 80)
            if new_vr != vr:
                obj["verdict_reason"] = new_vr
                changed = True

        sq = (obj.get("source_quality", "") or "").strip().lower()
        if sq not in {"high", "low"} and sq:
            obj["source_quality"] = "low"
            changed = True
    return arr, changed


def sanitize_block(block: str) -> Tuple[str, bool, int]:
    changed = False
    arr_s, arr_e = find_first_top_level_array(block)
    if arr_s is None:
        return block, False, 0
    arr_text_orig = block[arr_s:arr_e]
    try:
        arr = json.loads(arr_text_orig)
    except Exception:
        return block, False, 0

    arr, ch2 = _repair_doc_items(arr)
    if ch2:
        changed = True

    arr_canon = json.dumps(arr, ensure_ascii=False)
    head = block[:arr_s] + arr_canon
    tail = block[arr_e:]

    # normalize conflict line
    lines = tail.splitlines(keepends=False)
    i = 0
    while i < len(lines) and (not lines[i].strip() or PUNCT_ONLY.match(lines[i])):
        i += 1
    if i >= len(lines):
        new_tail = "\nNo conflict — Evidence aligns.\n"
        return head + new_tail, True, len(arr)

    conflict_idx = None
    ctype_canon, creason_raw = None, None
    pat = re.compile(rf"^\s*{CONFLICT_TYPES_RE}\s*{SEP_RE}\s*(.+?)\s*$", re.IGNORECASE)
    non_empty = [k for k in range(i, len(lines)) if lines[k].strip()]
    for k in non_empty[:3]:
        m = pat.match(lines[k])
        if m:
            ctype_canon = _normalize_conflict_type(m.group(1))
            if ctype_canon:
                creason_raw = m.group(2)
                conflict_idx = k
                break
    if conflict_idx is None:
        normalized = "No conflict — Evidence aligns."
        new_tail = "\n" + normalized + "\n" + "\n".join(lines[i:])
        return head + new_tail, True, len(arr)

    reason = DOC_RANGE.sub("multiple docs", creason_raw or "")
    reason = re.sub(r"\s+", " ", reason).strip()
    # v5: conflict_reason ≤ 50 words
    reason = _trim_words(reason, 50)
    normalized = f"{ctype_canon} — {reason}"
    rest_after_conflict = lines[conflict_idx + 1 :]
    new_tail_lines = [normalized] + rest_after_conflict
    new_tail = "\n" + "\n".join(new_tail_lines)

    if head + new_tail != block:
        changed = True
    return head + new_tail, changed, len(arr)


def _fix_final_oob_citations(final_text: str, max_id: int) -> Tuple[str, bool]:
    changed = False

    def repl(m):
        nonlocal changed
        n = int(m.group(1))
        if n < 1 or n > max_id:
            changed = True
            return ""  # drop out-of-bounds cite
        return m.group(0)

    fixed = CITE.sub(repl, final_text)
    return fixed, changed


def sanitize_record(rec: dict) -> Tuple[dict, bool]:
    raw = rec.get("raw", "")
    start, end, inner = extract_think_span(raw)
    if start is None:
        return rec, False
    new_block, ch1, max_id = sanitize_block(inner)
    tail = raw[end:]

    # also fix OOB citations in the FINAL answer area
    fixed_tail, ch2 = _fix_final_oob_citations(tail, max_id)
    if not (ch1 or ch2):
        return rec, False
    rec = dict(rec)
    rec["raw"] = raw[:start] + new_block + fixed_tail
    return rec, True


def main():
    ap = argparse.ArgumentParser()
    # accept both --in/--out and --in_jsonl/--out_jsonl
    ap.add_argument("--in", dest="in_jsonl", required=False)
    ap.add_argument("--out", dest="out_jsonl", required=False)
    ap.add_argument("--in_jsonl")
    ap.add_argument("--out_jsonl")
    args = ap.parse_args()

    in_p = args.in_jsonl or getattr(args, "in_jsonl", None) or getattr(args, "in", None)
    out_p = args.out_jsonl or getattr(args, "out_jsonl", None) or getattr(args, "out", None)
    if not in_p or not out_p:
        raise SystemExit("Provide --in and --out (or --in_jsonl/--out_jsonl)")

    out, changed = [], 0
    for rec in read_jsonl(in_p):
        new_rec, ch = sanitize_record(rec)
        if ch:
            changed += 1
        out.append(new_rec)
    write_jsonl(out_p, out)
    print(json.dumps({"total": len(out), "changed": changed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()