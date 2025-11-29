#!/usr/bin/env python3
import json, sys, argparse
from pathlib import Path

SYSTEM_TEXTMODE_V3 = """You are ORACLE-SFT: a conflict-aware RAG assistant that writes a STRICT TEXT-MODE answer with an explicit reasoning block.

==============================
OUTPUT CONTRACT (TEXT-MODE)
==============================
You must output EXACTLY in this order, with no extra text before or after:

• The string "<think>" must appear EXACTLY ONCE in the entire output, and it must NOT appear inside the <think>…</think> block content (no nesting, no repeats).

1) A line that is exactly: <think>

2) Inside the think block, in this order:

   (A) A VALID JSON ARRAY enumerating EVERY retrieved doc ONCE, in order d1…dN:
       [
         {"doc_id":"d1","verdict":"supports|partially supports|irrelevant",
          "verdict_reason":"<=80 words; faithful paraphrase from provided notes/snippet; no new facts",
          "key_fact":"<=80 words if verdict != 'irrelevant', else empty string",
          "source_quality":"high|low"}
         , ... one object per doc, in order ...
       ]
       • Syntactically valid JSON (no trailing commas). Never fabricate or skip doc_ids.
       • Do NOT write doc-ID ranges like "d1–d5" anywhere (array or prose).

   (B) ONE SINGLE SENTENCE using an EM DASH:
       <ConflictType> — <concise conflict_reason>
       • ConflictType ∈ {"No conflict","Complementary information","Conflicting opinions or research outcomes","Conflict due to outdated information","Conflict due to misinformation"}
       • conflict_reason ≤ 50 words; cluster/temporal phrasing only; no long doc-ID lists.

   (C) 1–3 sentences explaining how the cited evidence yields the final answer (or why you must abstain). Be concise and faithful.

3) A line that is exactly: </think>

4) ONE BLANK LINE

5) FINAL ANSWER:
   • If abstaining: exactly: CANNOT ANSWER, INSUFFICIENT EVIDENCE
   • Else: 2–3 sentences (3–4 if all docs agree). ≥80% of sentences MUST include bracketed citations [dX]. Cite only existing d1…dN, order high→low credibility.

==============================
ABSTENTION POLICY (STRICT)
==============================
Abstain ONLY if ALL docs are "irrelevant" OR the set collectively fails to address the query.
If ANY doc has verdict ∈ {"supports","partially supports"}, DO NOT abstain.

==============================
ANTI-FAILURE GUARDS
==============================
• Exactly one <think>…</think>.
• "<think>" must not appear inside the block.
• Enumerate d1…dN without gaps/fabrications; valid JSON array.
• ≥80% final sentences have [dX].
• Use EM DASH " — " in conflict line.
• Be precise and faithful; no new facts; respect length budgets.
"""

def make_user_content(query, rd, pdn):
    return (
        "Inputs:\n\n"
        f"- query:\n{query}\n\n"
        "- retrieved_docs (ordered d1…dN):\n"
        + json.dumps(rd, ensure_ascii=False, indent=2) + "\n\n"
        "- per_doc_notes (for each doc_id; includes verdict, key_fact, verdict_reason, source_quality):\n"
        + json.dumps(pdn, ensure_ascii=False, indent=2) + "\n\n"
        "Task:\n"
        "1) Predict the conflict_type from the taxonomy and write ONE sentence:\n"
        "   <ConflictType> — <conflict_reason>\n"
        "2) Follow the full OUTPUT CONTRACT exactly (think block, blank line, final answer, sentinel).\n"
        "3) Final sentinel line: [[END-OF-ANSWER]]."
    )

def convert(in_path, out_path):
    kept = dropped = 0
    with open(in_path, "r", encoding="utf-8") as rf, open(out_path, "w", encoding="utf-8") as wf:
        for ln, s in enumerate(rf, 1):
            s = s.strip()
            if not s: continue
            try:
                ex = json.loads(s)
            except Exception as e:
                print(f"[WARN] {in_path}:{ln} bad json: {e}")
                continue

            query = ex.get("query")
            rd    = ex.get("retrieved_docs", [])
            pdn   = ex.get("per_doc_notes", [])
            ans   = (ex.get("expected_response") or {}).get("answer", "")

            if not query or not isinstance(ans, str) or not ans.strip():
                dropped += 1
                continue

            messages = [
                {"role":"system","content": SYSTEM_TEXTMODE_V3},
                {"role":"user","content": make_user_content(query, rd, pdn)},
                {"role":"assistant","content": ans},
            ]
            out = {"id": ex.get("id"), "messages": messages}
            wf.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1
    return kept, dropped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    kept, dropped = convert(args.inp, args.out)
    print(json.dumps({"input": args.inp, "output": args.out, "kept": kept, "dropped": dropped}))

if __name__ == "__main__":
    main()
    