#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Normalize the ConflictType on the think-block label line:

Canonical set (your preference):
  - No conflict
  - Complementary information
  - Conflicting opinions or research outcomes
  - Conflict due to misinformation
  - Conflict due to outdated information

What this script does
---------------------
• For each of {train,val,test}.text.messages_v5.jsonl:
    - Parse each JSON row
    - Find the assistant message
    - Locate the single label line inside <think>…</think> that contains an EM DASH " — "
    - Normalize the *left* side (ConflictType) to one of the canonical strings above
      regardless of case, stray capitalization, “and” vs “or”, or minor variant wording.
    - Leave the right side (conflict_reason) untouched
    - Write to *_final.jsonl

No other edits are made.
"""

import argparse, json, os, re, sys

CANONICAL = {
    "no_conflict": "No conflict",
    "complementary": "Complementary information",
    "opinions_or_outcomes": "Conflicting opinions or research outcomes",
    "misinformation": "Conflict due to misinformation",
    "outdated": "Conflict due to outdated information",
}

def normalize_label(lhs_raw: str) -> str:
    """
    Map many free-form variants to the canonical label strings.
    We tolerate:
      • any capitalization
      • 'and'/'or'/'&'/'/' between opinions and research outcomes
      • stray 'due to' capitalization
      • common typos around that phrase
    """
    s = lhs_raw.strip()
    low = re.sub(r"\s+", " ", s.lower())

    # Fast exact/near-exact matches
    if "no conflict" in low:
        return CANONICAL["no_conflict"]

    # Complementary
    if "complementary" in low:
        # Handle the erroneous "conflict due to complementary information" by downgrading to Complementary info
        return CANONICAL["complementary"]

    # Misinformation
    if "misinformation" in low:
        return CANONICAL["misinformation"]

    # Outdated information
    if "outdated" in low or "out-dated" in low:
        return CANONICAL["outdated"]

    # Conflicting opinions or research outcomes (handle lots of variants)
    # Accept “conflicting/ conflict”, “opinions/ opinion/ opnions…”, connector, and “research outcome(s)”
    if re.search(r"\bconflict\w*\b", low) and \
       re.search(r"\bopin\w*\b", low) and \
       re.search(r"\bresearch\s+outcome\w*\b", low):
        return CANONICAL["opinions_or_outcomes"]

    # Also handle forms like "conflicting opinions and research outcomes", "conflicting opinions/research outcomes"
    if re.search(r"conflict\w*.*opin\w*.*(and|or|/|&).*research\s+outcome\w*", low):
        return CANONICAL["opinions_or_outcomes"]

    # If nothing matched, try some gentle fallbacks:

    # “conflicting … opinions … outcomes” (very loose)
    if "conflict" in low and "opin" in low and "outcome" in low:
        return CANONICAL["opinions_or_outcomes"]

    # Last-resort: return original (we won’t error; we’ll keep original so we don’t damage content)
    return lhs_raw.strip()

def extract_think_block(text: str):
    if "<think>" not in text or "</think>" not in text:
        return None
    start = text.index("<think>") + len("<think>")
    end = text.index("</think>")
    return start, end, text[start:end]

def normalize_in_assistant(content: str):
    """
    Find the label line within the think block and normalize only the left side
    of ' — ' to the canonical label.
    """
    got = extract_think_block(content)
    if not got:
        return content, False
    start, end, think_body = got

    lines = think_body.splitlines()
    changed = False

    # Find the FIRST line in the think block that contains the EM DASH " — "
    # (Your format has exactly one such line.)
    for i, line in enumerate(lines):
        if " — " in line:
            lhs, mdash, rhs = line.partition(" — ")
            new_lhs = normalize_label(lhs)
            if new_lhs != lhs.strip():
                lines[i] = f"{new_lhs} — {rhs}"
                changed = True
            break

    if not changed:
        return content, False

    new_think_body = "\n".join(lines)
    new_content = content[:start] + new_think_body + content[end:]
    return new_content, True

def process_file(in_path: str, out_path: str):
    total = 0
    modified = 0
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line.strip():
                fout.write(line + "\n")
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                # pass through unmodified
                fout.write(line + "\n")
                continue

            msgs = obj.get("messages", [])
            if not isinstance(msgs, list):
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            # Find assistant message
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "assistant" and "content" in m:
                    new_content, did = normalize_in_assistant(m["content"])
                    if did:
                        m["content"] = new_content
                        modified += 1
                    break

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return total, modified

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", required=True, help="Directory containing *.text.messages_v5.jsonl")
    args = ap.parse_args()

    in_files = [
        ("train.text.messages_v5.jsonl", "train.text.messages_v5_final.jsonl"),
        ("val.text.messages_v5.jsonl",   "val.text.messages_v5_final.jsonl"),
        ("test.text.messages_v5.jsonl",  "test.text.messages_v5_final.jsonl"),
    ]

    for src, dst in in_files:
        in_path  = os.path.join(args.splits_dir, src)
        out_path = os.path.join(args.splits_dir, dst)
        if not os.path.isfile(in_path):
            print(f"[WARN] missing: {in_path}")
            continue
        total, modified = process_file(in_path, out_path)
        print(f"[OK] {in_path} → {out_path}  (rows: {total}, label-lines normalized: {modified})")

if __name__ == "__main__":
    main()
    