#!/usr/bin/env python3
"""
pretty_print_textmode_v5.py
---------------------------
Pretty-print a single example by id, showing:

1) GOLD (from canon_jsonl, e.g. data/splits/val_v5.jsonl)
   - id
   - query
   - conflict_type
   - per_doc_notes
   - gold_answer (if present)
   - expected_response.answer (if present)

2) MODEL OUTPUT (from gens_jsonl, e.g. sanitized dev generations)
   - <think> ... </think> block pretty-printed:
       * JSON array indented
       * blank line before conflict line
       * blank line before explanation
   - final answer + [[END-OF-ANSWER]] printed as-is
   - NO headings before the model output.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln, s in enumerate(f, 1):
            s = s.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                raise ValueError(f"{path}:{ln} bad json: {e}")


def index_by_id(path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for ex in read_jsonl(path):
        cid = ex.get("id")
        if isinstance(cid, str):
            out[cid] = ex
    return out


def print_gold(ex: Dict[str, Any]):
    """Pretty-print gold information from the canon file."""
    cid = ex.get("id", "")
    query = ex.get("query", "")
    ctype = ex.get("conflict_type", "")

    print("=========== GOLD (CANON) ===========")
    print(f"ID: {cid}")
    print()
    print("Query:")
    print(f"  {query}")
    print()
    print(f"Gold conflict_type: {ctype}")
    print()
    print("Gold per_doc_notes:")

    notes = ex.get("per_doc_notes") or []
    if not notes:
        print("  (none)")
    else:
        for n in notes:
            did = n.get("doc_id", "")
            verdict = n.get("verdict", "")
            sq = n.get("source_quality", "")
            key_fact = n.get("key_fact", "")
            vr = n.get("verdict_reason", "")

            print(f"  - {did}:")
            print(f"      verdict        : {verdict}")
            print(f"      source_quality : {sq}")
            if key_fact:
                print(f"      key_fact       : {key_fact}")
            if vr:
                print(f"      verdict_reason : {vr}")
            print()

    gold_answer = ex.get("gold_answer")
    if gold_answer:
        print("Gold gold_answer:")
        print(f"  {gold_answer}")
        print()

    exp = ex.get("expected_response") or {}
    exp_answer = exp.get("answer", "")
    if exp_answer:
        print("Gold expected_response.answer:")
        print(f"  {exp_answer}")
    else:
        if exp.get("abstain"):
            print("Gold expected_response.answer:")
            print("  (abstain=True; no answer text)")
    print("====================================")


def _find_first_top_level_array(block: str):
    """Return (start_idx, end_idx) of first top-level JSON array, or (None, None)."""
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


def format_think_inner(inner: str) -> str:
    """
    Pretty-print the inside of <think>...</think>:

    - JSON array pretty-printed with indent=2
    - blank line
    - conflict line
    - blank line
    - explanation lines

    If parsing fails, returns the original inner text.
    """
    s = inner.strip("\n")
    if not s.strip():
        return s

    start, end = _find_first_top_level_array(s)
    if start is None:
        # no array, just return as-is
        return s

    arr_text = s[start:end]
    try:
        arr = json.loads(arr_text)
    except Exception:
        # cannot parse; return as-is
        return s

    pretty_arr = json.dumps(arr, ensure_ascii=False, indent=2)

    tail = s[end:].strip("\n")
    if not tail.strip():
        return pretty_arr

    lines = [ln.rstrip() for ln in tail.splitlines()]

    # First non-empty line after the array is treated as the conflict label line.
    conflict_line = None
    rest_lines = []
    first_seen = False
    for ln in lines:
        if not first_seen:
            if not ln.strip():
                continue
            conflict_line = ln
            first_seen = True
        else:
            rest_lines.append(ln)

    parts = [pretty_arr]
    if conflict_line:
        parts.append("")
        parts.append(conflict_line)
    if rest_lines:
        parts.append("")
        parts.extend(rest_lines)

    return "\n".join(parts)


def pretty_print_model_raw(raw: str):
    """
    Print model raw output with a pretty <think> block:
    - <think> on its own line
    - formatted inner content
    - </think>
    - then final answer tail as-is.
    """
    text = raw or ""
    start_tag = "<think>"
    end_tag = "</think>"

    m1 = text.find(start_tag)
    m2 = text.find(end_tag)

    if m1 == -1 or m2 == -1 or m2 <= m1:
        # fallback: just print raw as-is
        print(text)
        return

    before = text[:m1]
    inner = text[m1 + len(start_tag) : m2]
    after = text[m2 + len(end_tag) :]

    before = before.strip("\n")

    if before:
        print(before.rstrip("\n"))
        print()

    print("<think>")
    pretty_inner = format_think_inner(inner)
    if pretty_inner:
        print(pretty_inner.rstrip("\n"))
    print("</think>")

    after = after.lstrip("\n")
    if after:
        print()
        # print tail exactly as-is (final answer + sentinel)
        print(after, end="" if after.endswith("\n") else "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--canon_jsonl",
        required=True,
        help="Stage-3 split file with gold fields (e.g., data/splits/val_v5.jsonl)",
    )
    ap.add_argument(
        "--gens_jsonl",
        required=True,
        help="Sanitized generations JSONL with {'id','raw'} (e.g. dev.sanitized.jsonl)",
    )
    ap.add_argument(
        "--id",
        required=True,
        help="Example id to inspect (e.g. '#0198')",
    )
    args = ap.parse_args()

    if not Path(args.canon_jsonl).exists():
        raise SystemExit(f"canon_jsonl not found: {args.canon_jsonl}")
    if not Path(args.gens_jsonl).exists():
        raise SystemExit(f"gens_jsonl not found: {args.gens_jsonl}")

    canon_by_id = index_by_id(args.canon_jsonl)
    gens_by_id = index_by_id(args.gens_jsonl)

    cid = args.id
    if cid not in canon_by_id:
        raise SystemExit(f"id {cid!r} not found in canon_jsonl ({args.canon_jsonl})")
    if cid not in gens_by_id:
        raise SystemExit(f"id {cid!r} not found in gens_jsonl ({args.gens_jsonl})")

    gold_ex = canon_by_id[cid]
    gen_ex = gens_by_id[cid]

    # 1) GOLD
    print_gold(gold_ex)

    # 2) MODEL OUTPUT – no headings, just separation + pretty <think> block
    raw = gen_ex.get("raw", "")
    print("\n\n\n", end="")  # a few blank lines as separator
    pretty_print_model_raw(raw)


if __name__ == "__main__":
    main()