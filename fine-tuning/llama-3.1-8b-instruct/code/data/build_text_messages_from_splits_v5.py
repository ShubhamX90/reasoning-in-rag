#!/usr/bin/env python3
"""
Render *.text.messages_v5.jsonl (for train/dev) using:
  - system prompt file (system_text_mode_sft.v5.txt)
  - user template (user_text_mode_sft.v5.txt) with {query}, {retrieved_docs}, {per_doc_notes}
Assistant content is produced as:
  - If ex['think'] exists → use it verbatim, then blank line, final answer,
    then '[[END-OF-ANSWER]]'.
  - Else build a minimal, compliant <think> block from raw fields (no extra checks/drops),
    then the final answer + sentinel.
No other filtering is performed.
"""

import json, argparse
from pathlib import Path

THINK_OPEN  = "<think>"
THINK_CLOSE = "</think>"
SENTINEL    = "[[END-OF-ANSWER]]"
EM_DASH     = " — "  # ensure the true em dash

def load_text(p): 
    return Path(p).read_text(encoding="utf-8").strip()

def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for ln, s in enumerate(f, 1):
            s = s.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                raise ValueError(f"{p}:{ln} bad json: {e}")

def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def clamp_words(text: str, max_words: int):
    if not isinstance(text, str): return ""
    words = text.strip().split()
    return " ".join(words[:max_words]) if len(words) > max_words else text.strip()

def build_doc_array(per_doc_notes, retrieved_docs):
    """Keep order of retrieved_docs (d1..dN). Use notes when present; fill minimal fields otherwise."""
    note_by_id = {n.get("doc_id"): n for n in (per_doc_notes or [])}
    out = []
    for d in (retrieved_docs or []):
        did = d.get("doc_id")
        n   = note_by_id.get(did, {}) or {}
        verdict = (n.get("verdict") or "irrelevant").strip().lower()
        if verdict not in {"supports","partially supports","irrelevant"}:
            verdict = "irrelevant"
        key_fact = "" if verdict == "irrelevant" else (n.get("key_fact") or "")
        out.append({
            "doc_id": did or "",
            "verdict": verdict,
            "verdict_reason": clamp_words(n.get("verdict_reason",""), 80),
            "key_fact": clamp_words(key_fact, 80),
            "source_quality": ("high" if (n.get("source_quality","").strip().lower()=="high") else "low"),
        })
    return out

def render_user(user_template: str, ex: dict) -> str:
    return user_template.format(
        query=ex.get("query",""),
        retrieved_docs=json.dumps(ex.get("retrieved_docs",[]), ensure_ascii=False, indent=2),
        per_doc_notes=json.dumps(ex.get("per_doc_notes",[]), ensure_ascii=False, indent=2),
    )

def build_assistant_from_fields(ex: dict) -> str:
    # final answer line
    exp = ex.get("expected_response") or {}
    final_answer = "CANNOT ANSWER, INSUFFICIENT EVIDENCE" if exp.get("abstain") else (exp.get("answer") or "")
    # try to use provided think if present
    think_text = (ex.get("think") or "").strip()
    if think_text:
        return f"{think_text}\n\n{final_answer}\n{SENTINEL}"

    # minimal compliant think if not present, using raw fields (no drops)
    docs = build_doc_array(ex.get("per_doc_notes",[]), ex.get("retrieved_docs",[]))
    doc_json = json.dumps(docs, ensure_ascii=False)
    creason = clamp_words(ex.get("conflict_reason","").strip(), 50)
    ctype   = ex.get("conflict_type","").strip() or "No conflict"  # won't be empty post-filter in Split script
    label   = f"{ctype}{EM_DASH}{creason}" if creason else f"{ctype}{EM_DASH}Reason not provided"

    synth_line = "We integrate the provided snippets (prioritizing higher-quality sources) to reach the final answer."

    think = f"{THINK_OPEN}{doc_json},\n{creason or 'Reason not provided'},\n{label},\n{synth_line}\n{THINK_CLOSE}"
    return f"{think}\n\n{final_answer}\n{SENTINEL}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)   # e.g., data/splits/train.v5.jsonl
    ap.add_argument("--out_jsonl", required=True)  # e.g., data/splits/train.text.messages_v5.jsonl
    ap.add_argument("--system_prompt_path", required=True)
    ap.add_argument("--user_prompt_path", required=True)
    args = ap.parse_args()

    system_prompt = load_text(args.system_prompt_path)
    user_template = load_text(args.user_prompt_path)

    out_rows = []
    kept = 0
    for ex in read_jsonl(args.in_jsonl):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": render_user(user_template, ex)},
            {"role": "assistant", "content": build_assistant_from_fields(ex)},
        ]
        out_rows.append({"id": ex.get("id",""), "messages": messages})
        kept += 1

    write_jsonl(args.out_jsonl, out_rows)
    print(f"[Build] {args.in_jsonl} → {args.out_jsonl}  kept={kept}")

if __name__ == "__main__":
    main()