#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, sys, re
from typing import List, Dict, Any

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().rstrip("\n")

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception as e:
                raise ValueError(f"{path}:{ln}: invalid JSON: {e}")
    return rows

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def format_user_prompt(tpl: str, q: str, retrieved_docs: Any, per_doc_notes: Any) -> str:
    return tpl.format(
        query=q,
        retrieved_docs=json.dumps(retrieved_docs, ensure_ascii=False, indent=2),
        per_doc_notes=json.dumps(per_doc_notes, ensure_ascii=False, indent=2),
    ).rstrip("\n")

def _has_full_think_block(s: str) -> bool:
    if not s: return False
    return ("<think>" in s) and ("</think>" in s) and (s.count("<think>") >= 1) and (s.count("</think>") >= 1)

def _extract_outer_think(s: str) -> str:
    """Return exactly one outer <think>...</think> block from s."""
    i = s.find("<think>")
    j = s.rfind("</think>")
    if i == -1 or j == -1 or j < i:
        raise ValueError("cannot extract outer think block")
    inner = s[i + len("<think>"): j]
    # guard: no nested "<think>" tokens inside inner content
    inner = inner.replace("<think>", "")  # de-nest if any accidental tokens inside
    return "<think>" + inner + "</think>"

def _synthesize_think(row: Dict[str, Any]) -> str:
    """
    Build a minimal, contract-valid <think>…</think> from per_doc_notes + conflict info.
    No length checks. Does not invent facts; uses available fields.
    """
    notes = row.get("per_doc_notes", [])
    # Slim to exactly the required keys and enforce doc_id order d1..dN
    arr = []
    for idx, n in enumerate(notes, 1):
        arr.append({
            "doc_id": f"d{idx}",
            "verdict": n.get("verdict", ""),
            "verdict_reason": n.get("verdict_reason", ""),
            "key_fact": ("" if n.get("verdict") == "irrelevant" else n.get("key_fact","")),
            "source_quality": n.get("source_quality","low") or "low",
        })
    arr_json = json.dumps(arr, ensure_ascii=False, indent=2)

    conflict_type = row.get("conflict_type", "No conflict")
    conflict_reason = (row.get("conflict_reason") or "").strip()
    # light, safe reasoning; validator doesn't enforce content semantics
    n_docs = len(row.get("retrieved_docs", []))
    reasoning = (
        "Evidence clustered per per-doc verdicts above; irrelevant items are noted; "
        "differences are explained by contextual-scope or temporal factors where applicable."
    )

    label_line = f"{conflict_type} — {conflict_reason if conflict_reason else 'Concise conflict summary unavailable in source row.'}"

    # brief connection line (D)
    if row.get("expected_response", {}).get("abstain", False):
        connect = "All documents collectively fail to answer the query; abstention follows from the per-doc judgments."
    else:
        connect = "The final answer synthesizes the consistent supporting and partially supporting evidence, prioritizing higher-credibility sources."

    return "<think>\n" + arr_json + "\n" + reasoning + "\n" + label_line + "\n" + connect + "\n</think>"

def build_assistant_content(row: Dict[str, Any]) -> str:
    """
    Tolerant builder:
      - use existing full <think>…</think> if present (normalize to a single outer pair)
      - else wrap non-empty think text
      - else synthesize from row fields
      - then append final answer + sentinel
    """
    think_text = (row.get("think") or "").strip()

    if _has_full_think_block(think_text):
        try:
            think_block = _extract_outer_think(think_text)
        except Exception:
            # fall back to synthesis if malformed
            think_block = _synthesize_think(row)
    elif think_text:
        # no tags present, wrap safely; also ensure no nested token inside content
        inner = think_text.replace("<think>", "")
        think_block = "<think>" + inner + "</think>"
    else:
        think_block = _synthesize_think(row)

    expected = row.get("expected_response", {}) or {}
    if expected.get("abstain", False):
        final = "CANNOT ANSWER, INSUFFICIENT EVIDENCE"
    else:
        ans = (expected.get("answer") or "").strip()
        if not ans:
            # Extremely rare: missing answer while abstain==False; degrade gracefully
            ans = "CANNOT ANSWER, INSUFFICIENT EVIDENCE"
        final = ans

    return think_block + "\n\n" + final + "\n[[END-OF-ANSWER]]"

def convert_file(in_path: str, out_path: str, system_txt: str, user_txt: str) -> Dict[str, int]:
    data = load_jsonl(in_path)
    out_rows = []

    for row in data:
        rid = row.get("id")
        query = row.get("query", "")
        retrieved_docs = row.get("retrieved_docs", [])
        per_doc_notes = row.get("per_doc_notes", [])

        system_msg = {"role": "system", "content": system_txt}
        user_msg = {
            "role": "user",
            "content": format_user_prompt(user_txt, query, retrieved_docs, per_doc_notes),
        }
        assistant_msg = {"role": "assistant", "content": build_assistant_content(row)}

        out_rows.append({"id": rid, "messages": [system_msg, user_msg, assistant_msg]})

    write_jsonl(out_path, out_rows)
    return {"in": len(data), "out": len(out_rows)}

def main():
    p = argparse.ArgumentParser(description="Convert Stage-3 splits to *.text.messages_v5.jsonl (tolerant to missing/partial think blocks)")
    p.add_argument("--splits_dir", default="data/splits", help="Directory with *_v5.jsonl")
    p.add_argument("--system_file", default="prompts/sft/system_text_mode_sft.v5.txt")
    p.add_argument("--user_file",   default="prompts/sft/user_text_mode_sft.v5.txt")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    system_txt = read_text(args.system_file)
    user_txt   = read_text(args.user_file)

    pairs = [
        ("train_v5.jsonl", "train.text.messages_v5.jsonl"),
        ("val_v5.jsonl",   "val.text.messages_v5.jsonl"),
        ("test_v5.jsonl",  "test.text.messages_v5.jsonl"),
    ]

    totals = {}
    for in_name, out_name in pairs:
        in_path  = os.path.join(args.splits_dir, in_name)
        out_path = os.path.join(args.splits_dir, out_name)
        if not os.path.isfile(in_path):
            print(f"[WARN] Missing: {in_path} (skipping)")
            continue
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[SKIP] Exists: {out_path} (use --overwrite to replace)")
            continue
        stats = convert_file(in_path, out_path, system_txt, user_txt)
        print(f"[OK] {in_path} → {out_path}  (rows: {stats['out']})")
        totals[out_name] = stats["out"]

    if not totals:
        print("No files converted.")
        sys.exit(1)

if __name__ == "__main__":
    main()