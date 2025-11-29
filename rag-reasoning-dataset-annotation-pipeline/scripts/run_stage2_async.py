"""
Stage-2 Async: Conflict-Aware Reasoning Aggregation
--------------------------------------------------
Processes multiple queries concurrently to produce Stage-2 reasoning outputs.

Each record contains:
  • the query
  • the gold conflict_type
  • per-document notes (from Stage-1)

Outputs a concise, evidence-grounded macro-reasoning JSON for each query.

Usage:
  python -m scripts.run_stage2_async \
    --input ~/Desktop/rag-reasoning-dataset/data/stage1_outputs/stage1_final.jsonl \
    --output ~/Desktop/rag-reasoning-dataset/data/stage2_outputs/stage2_final.jsonl \
    --model gpt-4o-mini \
    --concurrency 12 \
    --limit 20
"""

import os
import json
import argparse
import asyncio
from pathlib import Path
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from src.utils import extract_json  # robust extractor we added

# ---------------------------------------------------------
# Prompt paths
# ---------------------------------------------------------
SYSTEM_PROMPT_PATH = str(Path.home() / "Desktop/rag-reasoning-dataset/prompts/system_stage2.txt")
USER_PROMPT_PATH = str(Path.home() / "Desktop/rag-reasoning-dataset/prompts/user_stage2.txt")

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def _load_api_key() -> str:
    key_path = os.path.expanduser("~/.openai_key")
    if os.path.exists(key_path):
        return open(key_path).read().strip()
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    raise ValueError("No API key found. Set OPENAI_API_KEY or create ~/.openai_key")

def _brace_safe_fill(template: str, mapping: dict) -> str:
    """Safely fills placeholders like {QUERY} etc. while preserving braces."""
    temp = template
    for k in mapping.keys():
        temp = temp.replace("{" + k + "}", f"@@{k}@@")
    temp = temp.replace("{", "{{").replace("}", "}}")
    for k, v in mapping.items():
        temp = temp.replace(f"@@{k}@@", "" if v is None else str(v))
    return temp

# ---------------------------------------------------------
# Async per-record processor
# ---------------------------------------------------------
async def process_record(client, semaphore, record, out_lock, fout_path, system_prompt, user_template, model_name):
    async with semaphore:
        rec_id = record.get("id", "")
        query = record.get("query", "")
        conflict_type = record.get("conflict_type", "UNKNOWN")
        per_doc_notes = record.get("per_doc_notes", [])

        # Fill user prompt
        user_prompt = _brace_safe_fill(
            user_template,
            {
                "QUERY": query,
                "CONFLICT_TYPE": conflict_type,
                "PER_DOC_NOTES_JSON": json.dumps(per_doc_notes, ensure_ascii=False, indent=2),
            },
        )

        # Retry loop with exponential backoff
        attempt, backoff = 0, 1.5
        while True:
            attempt += 1
            try:
                resp = await client.chat.completions.create(
                    model=model_name,
                    temperature=0.2,
                    max_tokens=250,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    timeout=90.0,
                    extra_headers={"Idempotency-Key": f"{rec_id}:{attempt}"},
                )

                content = (resp.choices[0].message.content or "").strip()
                parsed, frag = extract_json(content)

                if (
                    parsed
                    and isinstance(parsed, dict)
                    and "conflict_reason" in parsed
                    and "answerable_under_evidence" in parsed
                ):
                    record.update(parsed)
                else:
                    record.update({
                        "conflict_reason": "Failed to parse valid JSON.",
                        "answerable_under_evidence": False,
                    })
                break

            except Exception as e:
                if attempt >= 4:
                    record.update({
                        "conflict_reason": f"Exception after retries: {e}",
                        "answerable_under_evidence": False,
                    })
                    break
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 20.0)

        # Write each record safely
        async with out_lock:
            with open(fout_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

# ---------------------------------------------------------
# Main async runner
# ---------------------------------------------------------
async def run_stage2_async(input_path: str, output_path: str, limit: int | None, concurrency: int, model_name: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(api_key=_load_api_key())
    semaphore = asyncio.Semaphore(concurrency)
    out_lock = asyncio.Lock()

    system_prompt = load_text(SYSTEM_PROMPT_PATH)
    user_template = load_text(USER_PROMPT_PATH)

    tasks = []
    with open(input_path, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            if limit is not None and idx >= limit:
                break
            record = json.loads(line)
            tasks.append(process_record(
                client, semaphore, record, out_lock, output_path,
                system_prompt, user_template, model_name
            ))

    for f in tqdm_asyncio.as_completed(tasks, desc="Stage-2 async records", total=len(tasks)):
        await f

    print(f"\n✅ Stage-2 async completed using model: {model_name}\nOutput saved to: {output_path}")

# ---------------------------------------------------------
# CLI entry
# ---------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Async Stage-2 macro reasoning synthesis.")
    ap.add_argument("--input", required=True, help="Path to stage1_final.jsonl")
    ap.add_argument("--output", required=True, help="Path to stage2_final.jsonl")
    ap.add_argument("--model", required=True, help="OpenAI model name, e.g. gpt-4o-mini or gpt-4-turbo")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of records")
    ap.add_argument("--concurrency", type=int, default=12, help="Number of concurrent tasks")
    args = ap.parse_args()

    asyncio.run(run_stage2_async(args.input, args.output, args.limit, args.concurrency, args.model))