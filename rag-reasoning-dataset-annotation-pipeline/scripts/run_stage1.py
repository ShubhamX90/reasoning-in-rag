#!/usr/bin/env python3
"""
scripts/run_stage1.py

Run Stage-1 for a SINGLE query (all its retrieved docs) selected from
data/normalized/conflicts_normalized.jsoln, and write outputs to:
  data/stage1_outputs/stage1_q000X.jsonl

Examples:
  python scripts/run_stage1.py --qid 2
  python scripts/run_stage1.py --qid "#0002" --validate
  python scripts/run_stage1.py --qid 2 --normalized /abs/path/conflicts_normalized.jsoln
  python scripts/run_stage1.py --qid 2 --system prompts/system_stage1.txt --user prompts/user_stage1.txt

Notes:
- Defaults now point to rag-reasoning-dataset/prompts/{system,user}_stage1.txt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from openai import OpenAI

# ---------- Resolve project layout ----------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]          # rag-reasoning-dataset/
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
DATA_DIR = PROJECT_ROOT / "data"

# Ensure we can import src/utils.py
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import (  # type: ignore
    coerce_query_id,
    load_query_entry,
    to_stage1_items_from_entry,
    output_path_stage1,
    validate_stage1_record,
    ensure_parent,
)

# ---------- Helpers ----------
def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def make_user_prompt(tpl: str, item: Dict[str, Any]) -> str:
    repl = {
        "{QUERY}": item["query"],
        "{DOC_ID}": item["doc"]["doc_id"],
        "{URL}": item["doc"].get("source_url", "") or "",
        "{TEXT}": item["doc"].get("snippet", "") or "",
        "{TIMESTAMP}": item["doc"].get("timestamp", "") or "",
    }
    s = tpl
    for k, v in repl.items():
        s = s.replace(k, json.dumps(v)[1:-1] if isinstance(v, str) else str(v))
    return s


def call_stage1(client: OpenAI, system_prompt: str, user_prompt: str, model: str, temperature: float) -> Dict[str, Any]:
    # Handle GPT-5 models (no temperature parameter supported)
    if model.startswith("gpt-5"):
        r = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    else:
        r = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    return json.loads(r.choices[0].message.content)


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", required=True, help="Query id: 2, 0002, #0002, etc.")
    ap.add_argument("--normalized", default=None,
                    help="Path to normalized dataset (JSONL/JSOLN or JSON array). Default: data/normalized/conflicts_normalized.jsoln")
    ap.add_argument("--data-dir", default=None, help="Base data dir for outputs. Default: ./data")
    ap.add_argument("--system", default=None, help="Stage-1 system prompt. Default: prompts/system_stage1.txt")
    ap.add_argument("--user", default=None, help="Stage-1 user prompt. Default: prompts/user_stage1.txt")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--validate", action="store_true", help="Attach _validation_errors if any")
    ap.add_argument("--output", help="Override output path; default uses data/stage1_outputs/stage1_q000X.jsonl")
    args = ap.parse_args()

    normalized_path = Path(args.normalized) if args.normalized else (DATA_DIR / "normalized" / "conflicts_normalized.jsoln")
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    system_path = Path(args.system) if args.system else (PROMPTS_DIR / "system_stage1.txt")
    user_path = Path(args.user) if args.user else (PROMPTS_DIR / "user_stage1.txt")

    qid_norm = coerce_query_id(args.qid)
    entry = load_query_entry(normalized_path, qid_norm)
    items = to_stage1_items_from_entry(entry)

    system_prompt = load_text(system_path)
    user_tpl = load_text(user_path)

    client = OpenAI()
    outputs: List[Dict[str, Any]] = []

    for it in items:
        up = make_user_prompt(user_tpl, it)
        resp = call_stage1(client, system_prompt, up, model=args.model, temperature=args.temperature)
        resp["doc_id"] = resp.get("doc_id") or it["doc"]["doc_id"]
        if args.validate:
            errs = validate_stage1_record(resp)
            if errs:
                resp["_validation_errors"] = errs
        outputs.append(resp)

    out_path = Path(args.output) if args.output else output_path_stage1(data_dir, qid_norm)
    ensure_parent(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "query_id": qid_norm,
        "query": entry["query"],
        "num_docs": len(items),
        "output_file": str(out_path),
        "normalized_used": str(normalized_path),
        "system_prompt": str(system_path),
        "user_prompt": str(user_path),
    }
    mf_path = Path(str(out_path) + ".meta.json")
    mf_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()