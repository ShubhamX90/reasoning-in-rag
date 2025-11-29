"""
Stage-3 (Single Query Mode): Grounded Expected Response Generation
------------------------------------------------------------------
Runs Stage-3 synthesis for ONE specific query ID from Stage-2 JSONL.

Usage:
  python -m scripts.run_stage3 \
    --system_prompt ~/Desktop/rag-reasoning-dataset/prompts/system_stage3.txt \
    --user_prompt ~/Desktop/rag-reasoning-dataset/prompts/user_stage3.txt \
    --model gpt-5-chat-latest \
    --id "#0021"

It will:
  • Locate that record in the Stage-2 JSONL file
  • Send all retrieved docs + per_doc_notes + metadata to the LLM
  • Save output as: ~/Desktop/rag-reasoning-dataset/data/stage3_outputs/stage3_#0021.jsonl
"""

import os
import re
import json
import argparse
from pathlib import Path
from openai import OpenAI


# ============================================================
# HARD-CODED PATHS  (edit as needed)
# ============================================================

INPUT_PATH = os.path.expanduser(
    "~/Desktop/rag-reasoning-dataset/data/stage2_outputs/stage2_final.jsonl"
)
OUTPUT_DIR = os.path.expanduser(
    "~/Desktop/rag-reasoning-dataset/data/stage3_outputs/"
)


# ============================================================
# Helper Functions
# ============================================================

def read_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_api_key() -> str:
    key_path = os.path.expanduser("~/.openai_key")
    if os.path.exists(key_path):
        return open(key_path).read().strip()
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key.strip()
    raise ValueError("No API key found. Please set OPENAI_API_KEY or create ~/.openai_key")


def brace_safe_format(template: str, mapping: dict) -> str:
    """Safely substitute placeholders without breaking braces."""
    temp = template.replace("{", "{{").replace("}", "}}")
    for k in mapping.keys():
        temp = temp.replace(f"{{{{{k}}}}}", f"{{{k}}}")
    return temp.format(**mapping)


# ============================================================
# JSON Parsing Helpers
# ============================================================

def _strip_md_fences(txt: str) -> str:
    return re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.I | re.M)


def _longest_balanced_json(txt: str) -> str | None:
    stack, start = 0, None
    for i, ch in enumerate(txt):
        if ch == "{":
            stack += 1
            if start is None:
                start = i
        elif ch == "}":
            stack -= 1
            if stack == 0 and start is not None:
                return txt[start:i + 1]
    return None


def try_parse_json(raw: str):
    raw = raw.strip()
    if not raw:
        return None
    for attempt in [raw, _strip_md_fences(raw)]:
        try:
            return json.loads(attempt)
        except Exception:
            pass
    candidate = _longest_balanced_json(raw)
    if candidate:
        try:
            return json.loads(candidate)
        except Exception:
            pass
    cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return json.loads(cleaned)
    except Exception:
        return None


# ============================================================
# Model Call
# ============================================================

def call_model(client, model, system_prompt, user_prompt,
               temperature=0.0, max_tokens=2500):
    """Call OpenAI model and robustly extract JSON."""
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = (resp.choices[0].message.content or "").strip()
        parsed = try_parse_json(content)
        if parsed and isinstance(parsed, dict):
            return parsed
        return {"error": "json_parse_failure", "raw": content}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# Stage-3 Runner (Single Record)
# ============================================================

def run_stage3_single(system_prompt_path, user_prompt_path,
                      model, record_id, temperature=0.0):

    client = OpenAI(api_key=load_api_key())
    system_prompt = read_text(system_prompt_path)
    user_template = read_text(user_prompt_path)

    # locate record
    target_record = None
    with open(INPUT_PATH, "r", encoding="utf-8") as fin:
        for line in fin:
            record = json.loads(line)
            if record.get("id") == record_id:
                target_record = record
                break

    if not target_record:
        raise ValueError(f"Record with id {record_id} not found in {INPUT_PATH}")

    num_docs = len(target_record.get("retrieved_docs", []))
    print(f"Found record {record_id} with {num_docs} retrieved documents.")

    # fill user prompt
    user_prompt = brace_safe_format(user_template, {
        "query": target_record.get("query", ""),
        "retrieved_docs": json.dumps(target_record.get("retrieved_docs", []),
                                    ensure_ascii=False, indent=2),
        "per_doc_notes": json.dumps(target_record.get("per_doc_notes", []),
                                    ensure_ascii=False, indent=2),
        "conflict_type": target_record.get("conflict_type", ""),
        "conflict_reason": target_record.get("conflict_reason", ""),
        "answerable_under_evidence": str(target_record.get("answerable_under_evidence", True)).lower(),
        "gold_answer": target_record.get("gold_answer", "") or "",
    })

    print("\n--- Sending to model ---")
    print(f"Query: {target_record.get('query')}")
    print(f"Conflict type: {target_record.get('conflict_type')}")
    print(f"Answerable: {target_record.get('answerable_under_evidence')}")
    print("Calling model...")

    # model call
    out = call_model(client, model, system_prompt, user_prompt,
                     temperature=temperature, max_tokens=2500)

    # attach outputs
    if "error" in out:
        target_record["expected_response"] = {
            "answer": "CANNOT ANSWER, PARSE FAILURE",
            "evidence": [],
            "abstain": True,
            "abstain_reason": f"Model output error: {out['error']}"
        }
        target_record["think"] = ""
    else:
        target_record["expected_response"] = out.get("expected_response", out)
        target_record["think"] = out.get("think", "")

    # output file auto-naming
    output_file = Path(OUTPUT_DIR) / f"stage3_{record_id}.jsonl"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as fout:
        json.dump(target_record, fout, ensure_ascii=False)
        fout.write("\n")

    print("\n✅ Stage-3 completed successfully.")
    print(f"→ Output saved to: {output_file}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run Stage-3 for a single record by ID.")
    parser.add_argument("--system_prompt", required=True)
    parser.add_argument("--user_prompt", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--id", required=True, help="Record ID")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    run_stage3_single(
        system_prompt_path=args.system_prompt,
        user_prompt_path=args.user_prompt,
        model=args.model,
        record_id=args.id,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()