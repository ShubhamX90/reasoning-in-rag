"""
Stage-2 Single Query Runner
---------------------------
Fetches one query (by ID) from data/stage1_outputs/stage1_final.jsonl,
runs conflict reasoning using Stage-2 prompts, and writes output
to data/stage2_outputs/stage2_<id>.jsonl.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

# --- ensure src/ is importable ---
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils import extract_first_json as extract_json  # ✅


# ----------------------------- CONFIG -----------------------------
INPUT_PATH = "data/stage1_outputs/stage1_final.jsonl"
OUTPUT_DIR = "data/stage2_outputs"
SYSTEM_PROMPT_PATH = "prompts/system_stage2.txt"
USER_PROMPT_TEMPLATE_PATH = "prompts/user_stage2.txt"

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"❌ Input file not found at {INPUT_PATH}")


# ----------------------------- HELPERS -----------------------------
def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_api_key() -> str:
    """Load OpenAI API key from ~/.openai_key or environment variable."""
    key_path = os.path.expanduser("~/.openai_key")
    if os.path.exists(key_path):
        with open(key_path, "r") as f:
            return f.read().strip()
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    raise ValueError("No API key found. Set OPENAI_API_KEY or create ~/.openai_key")


def _brace_safe_fill(template: str, mapping: dict) -> str:
    """Safely fill placeholders like {QUERY} without breaking braces."""
    temp = template
    for k in mapping.keys():
        temp = temp.replace("{" + k + "}", f"@@{k}@@")
    temp = temp.replace("{", "{{").replace("}", "}}")
    for k, v in mapping.items():
        if v is None:
            v = ""
        temp = temp.replace(f"@@{k}@@", str(v))
    return temp


# ----------------------------- MODEL CALL -----------------------------
def call_stage2_model(user_prompt: str, system_prompt: str, model_name: str, retries: int = 1):
    """Call the model and ensure valid JSON output."""
    client = OpenAI(api_key=_load_api_key())
    for attempt in range(retries + 1):
        try:
            print(f"🧠 Calling model: {model_name} (attempt {attempt+1})")
            resp = client.chat.completions.create(
                model=model_name,
                temperature=0.2,
                max_tokens=350,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = (resp.choices[0].message.content or "").strip()
            parsed, _ = extract_json(content)

            if (
                parsed
                and isinstance(parsed, dict)
                and "conflict_reason" in parsed
                and "answerable_under_evidence" in parsed
            ):
                return parsed, False

            if attempt < retries:
                print(f"[RETRY {attempt+1}] Bad JSON, retrying...")
                continue

            return {
                "conflict_reason": "Failed to parse JSON.",
                "answerable_under_evidence": False,
            }, True

        except Exception as e:
            if attempt < retries:
                print(f"[RETRY {attempt+1}] Exception: {e}")
                continue
            return {
                "conflict_reason": f"Exception: {e}",
                "answerable_under_evidence": False,
            }, True


# ----------------------------- MAIN FUNCTION -----------------------------
def run_stage2_single(query_id: str, model_name: str):
    """Run Stage-2 reasoning for one query ID."""
    print(f"🔍 Looking for record with id={query_id} in {INPUT_PATH} ...")

    with open(INPUT_PATH, "r", encoding="utf-8") as fin:
        record = None
        for line in tqdm(fin, desc="Scanning records", ncols=90):
            obj = json.loads(line)
            if obj.get("id") == query_id:
                record = obj
                break

    if not record:
        raise ValueError(f"No record found with id={query_id} in {INPUT_PATH}")

    print(f"✅ Found query: {record['query']}\n")

    # Load prompts
    system_prompt = load_text(SYSTEM_PROMPT_PATH)
    user_template = load_text(USER_PROMPT_TEMPLATE_PATH)

    query = record.get("query", "")
    conflict_type = record.get("conflict_type", "UNKNOWN")
    per_doc_notes = record.get("per_doc_notes", [])

    # Fill user prompt safely
    user_prompt = _brace_safe_fill(
        user_template,
        {
            "QUERY": query,
            "CONFLICT_TYPE": conflict_type,
            "PER_DOC_NOTES_JSON": json.dumps(per_doc_notes, ensure_ascii=False, indent=2),
        },
    )

    # Call model
    result, retried = call_stage2_model(user_prompt, system_prompt, model_name, retries=1)

    # Merge with original record (explicit schema to preserve structure)
    final_record = {
        "id": record.get("id"),
        "query": record.get("query"),
        "conflict_type": record.get("conflict_type"),  # copied exactly from input
        "gold_answer": record.get("gold_answer", ""),
        "retrieved_docs": record.get("retrieved_docs", []),
        "per_doc_notes": record.get("per_doc_notes", []),
        "conflict_reason": result.get("conflict_reason", ""),
        "answerable_under_evidence": result.get("answerable_under_evidence", False),
    }

    # Save output
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    safe_id = query_id.replace("#", "")
    output_path = os.path.join(OUTPUT_DIR, f"stage2_{safe_id}.jsonl")

    with open(output_path, "w", encoding="utf-8") as fout:
        fout.write(json.dumps(final_record, ensure_ascii=False) + "\n")

    print(f"\n💾 Output written to: {output_path}")
    if retried:
        print("⚠️ Model output required retry or fallback.")
    else:
        print("✅ Stage-2 single query completed successfully.")


# ----------------------------- CLI -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage-2 reasoning for a single query.")
    parser.add_argument("--id", required=True, help="Query ID (e.g., #0177)")
    parser.add_argument("--model", required=True, help="Model name (e.g., gpt-5-chat-latest)")
    args = parser.parse_args()

    run_stage2_single(args.id, args.model)