#!/usr/bin/env python3
"""
Normalize ConflictType casing in *.v5.1.text.messages_v5_final.jsonl.

- Reads:  data/splits/{train,val,test}_v5.1.text.messages_v5_final.jsonl
- Writes: data/splits/{split}_v5.1.text.messages_v5_final.normalized.jsonl

It:
  * detects the <ConflictType> — <reason> line AFTER the JSON array
  * parses the left side (ConflictType) case-insensitively
  * rewrites it to the canonical string from CONFLICT_TYPES
"""

import json
from pathlib import Path
import re

CONFLICT_TYPES = [
    "No conflict",
    "Complementary information",
    "Conflicting opinions or research outcomes",
    "Conflict due to outdated information",
    "Conflict due to misinformation",
]
TYPE2IDX = {t: i for i, t in enumerate(CONFLICT_TYPES)}
TYPE2IDX_NORM = {t.lower(): i for t, i in TYPE2IDX.items()}

DASH_SEPS = (" — ", " – ", " - ")


def _span_after_first_top_level_json_array(text: str):
    start = text.find("[")
    if start < 0:
        return (None, None)
    depth, end, in_str, esc = 0, None, False, False
    for i in range(start, len(text)):
        ch = text[i]
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
                    end = i + 1
                    break
    return (start, end) if end is not None else (None, None)


def find_conflict_line_span(text: str):
    """Return (start_idx, end_idx) for the label line, or (None, None)."""
    _, array_end = _span_after_first_top_level_json_array(text)
    if array_end is None:
        return (None, None)
    tail = (text[array_end:] or "").lstrip("\n")
    offset0 = len(text) - len(tail)
    for ln in tail.splitlines():
        line = ln.strip()
        if not line:
            offset0 += len(ln) + 1
            continue
        for sep in DASH_SEPS:
            if sep in line:
                left = line.split(sep, 1)[0].strip()
                if left.lower() in TYPE2IDX_NORM:
                    s = text.find(line, offset0)
                    if s >= 0:
                        return (s, s + len(line))
        offset0 += len(ln) + 1
    return (None, None)


def normalize_conflict_line(text: str):
    """Return (new_text, changed_flag)."""
    ls, le = find_conflict_line_span(text)
    if ls is None:
        return text, False

    line = text[ls:le]
    for sep in DASH_SEPS:
        if sep in line:
            left, right = line.split(sep, 1)
            left_clean = left.strip()
            idx = TYPE2IDX_NORM.get(left_clean.lower())
            if idx is None:
                return text, False
            canonical = CONFLICT_TYPES[idx]
            new_line = f"{canonical}{sep}{right.lstrip()}"
            return text[:ls] + new_line + text[le:], (new_line != line)
    return text, False


def process_split(split: str):
    in_path = Path(f"data/splits/{split}_v5.1.text.messages_v5_final.jsonl")
    out_path = Path(f"data/splits/{split}_v5.1.text.messages_v5_final.normalized.jsonl")

    if not in_path.is_file():
        print(f"[WARN] Input missing for split={split}: {in_path}")
        return

    total = changed = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            ex = json.loads(line)
            msgs = ex.get("messages", [])
            if len(msgs) != 3 or msgs[2].get("role") != "assistant":
                fout.write(line)
                continue
            total += 1
            asst = msgs[2].get("content", "")
            new_asst, did = normalize_conflict_line(asst)
            if did:
                changed += 1
            msgs[2]["content"] = new_asst
            ex["messages"] = msgs
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[Split {split}] processed {total} assistant messages, changed {changed}, wrote → {out_path}")


def main():
    for split in ["train", "val", "test"]:
        process_split(split)


if __name__ == "__main__":
    main()
    