#!/usr/bin/env python3
import json, argparse, sys
from pathlib import Path

def each_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for ln, s in enumerate(f, 1):
            s=s.strip()
            if not s: continue
            try: yield json.loads(s)
            except Exception as e:
                raise ValueError(f"{p}:{ln} bad json: {e}")

def wrap_item(ex, sys_txt, user_tmpl):
    rd  = ex.get("retrieved_docs", [])
    pdn = ex.get("per_doc_notes", [])
    user = user_tmpl.format(
        query=ex.get("query",""),
        retrieved_docs=json.dumps(rd, ensure_ascii=False, indent=2),
        per_doc_notes=json.dumps(pdn, ensure_ascii=False, indent=2),
    )
    return {
        "id": ex.get("id"),
        "messages": [
            {"role": "system", "content": sys_txt},
            {"role": "user",   "content": user}
        ]
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_in", required=True)     # data/splits/dev.jsonl
    ap.add_argument("--system_prompt", required=True)
    ap.add_argument("--user_prompt", required=True)
    ap.add_argument("--out_jsonl", required=True)    # data/splits/dev.text.messages.jsonl
    args = ap.parse_args()

    sys_txt = Path(args.system_prompt).read_text(encoding="utf-8")
    usr_txt = Path(args.user_prompt).read_text(encoding="utf-8")

    items = list(each_jsonl(args.split_in))
    with open(args.out_jsonl, "w", encoding="utf-8") as wf:
        for ex in items:
            rec = wrap_item(ex, sys_txt, usr_txt)
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote → {args.out_jsonl} (count={len(items)})")

if __name__ == "__main__":
    main()