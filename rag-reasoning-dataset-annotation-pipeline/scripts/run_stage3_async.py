"""
Stage-3 Async: Grounded Synthesis & Expected Response Generation
---------------------------------------------------------------
Async version for faster parallel synthesis across queries.

Input, output, and prompt paths are fully configurable via CLI.
"""

"""
How to run:

python -m scripts.run_stage3_async \
  --input data/stage2_outputs/refusals11_stage2.jsonl \
  --output data/stage3_outputs/refusals11_stage3.jsonl \
  --system_prompt prompts/system_stage3.txt \
  --user_prompt prompts/user_stage3.txt \
  --model gpt-5-chat-latest \
  --temperature 0.0 \
  --concurrency 12

"""

import os
import re
import json
import asyncio
from pathlib import Path
from tqdm import tqdm
from openai import AsyncOpenAI, OpenAIError

# ============================================================
# Helpers
# ============================================================

def read_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_api_key() -> str:
    key_path = os.path.expanduser("~/.openai_key")
    if os.path.exists(key_path):
        return open(key_path).read().strip()
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    raise ValueError("No API key found. Set OPENAI_API_KEY or create ~/.openai_key")

def brace_safe_format(template: str, mapping: dict) -> str:
    """Safely format a template string that already contains braces."""
    temp = template.replace("{", "{{").replace("}", "}}")
    for k in mapping.keys():
        temp = temp.replace(f"{{{{{k}}}}}", f"{{{k}}}")
    return temp.format(**mapping)

# ============================================================
# JSON Extraction
# ============================================================

def try_parse_json(raw: str):
    raw = raw.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.I | re.M)
    try:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            frag = raw[start:end + 1]
            frag = re.sub(r",\s*([\]}])", r"\1", frag)
            return json.loads(frag)
    except Exception:
        pass
    return None

# ============================================================
# Async Model Call
# ============================================================

async def call_model_async(client, model, system_prompt, user_prompt,
                           temperature=0.0, max_tokens=2500, retries=3):
    """Calls the model asynchronously with JSON response enforcement."""
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = (resp.choices[0].message.content or "").strip()
            parsed = try_parse_json(content)
            if parsed and isinstance(parsed, dict):
                return parsed
            return {"error": "JSON parse failure", "raw": content}
        except OpenAIError as e:
            if attempt < retries - 1:
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            return {"error": str(e)}
        except Exception as e:
            return {"error": str(e)}

# ============================================================
# Record Processor
# ============================================================

async def process_record_async(client, record, system_prompt, user_template,
                               model, temperature):
    rec_id = record.get("id", "unknown")

    user_prompt = brace_safe_format(user_template, {
        "query": record.get("query", ""),
        "per_doc_notes": json.dumps(record.get("per_doc_notes", []), ensure_ascii=False, indent=2),
        "conflict_type": record.get("conflict_type", ""),
        "conflict_reason": record.get("conflict_reason", ""),
        "expected_behavior": record.get("expected_behavior", ""),
        "answerable_under_evidence": str(record.get("answerable_under_evidence", True)).lower(),
        "gold_answer": record.get("gold_answer", "") or "",
        "ranked_doc_ids": ", ".join(
            [n.get("doc_id") for n in record.get("per_doc_notes", []) if n.get("verdict") != "irrelevant"]
        )
    })

    out = await call_model_async(client, model, system_prompt, user_prompt,
                                 temperature=temperature)

    record["expected_response"] = out.get("expected_response", out)
    record["think"] = out.get("think", "")

    status = "answered"
    if "error" in out:
        status = "failed"
    elif isinstance(out.get("expected_response"), dict) and out["expected_response"].get("abstain", False):
        status = "abstained"

    return record, status

# ============================================================
# Async Runner
# ============================================================

async def run_stage3_async(input_path, output_path, system_prompt_path, user_prompt_path,
                           model="gpt-4o", temperature=0.0,
                           limit=None, concurrency=10):

    api_key = load_api_key()
    client = AsyncOpenAI(api_key=api_key)

    system_prompt = read_text(system_prompt_path)
    user_template = read_text(user_prompt_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "answered": 0, "abstained": 0, "failed": 0}

    with open(input_path, "r", encoding="utf-8") as fin:
        records = [json.loads(l) for l in fin]
    if limit:
        records = records[:limit]

    sem = asyncio.Semaphore(concurrency)

    async def sem_task(rec):
        async with sem:
            return await process_record_async(client, rec, system_prompt, user_template, model, temperature)

    tasks = [sem_task(rec) for rec in records]

    with open(output_path, "w", encoding="utf-8") as fout:
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                         desc="Stage-3 async", ncols=100):
            record, status = await coro
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["total"] += 1
            stats[status] = stats.get(status, 0) + 1

    summary_path = Path(output_path).with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n✅ Async Stage-3 completed.")
    print(f"Output → {output_path}")
    print("Summary:", json.dumps(stats, indent=2))

# ============================================================
# CLI Entrypoint
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Async Stage-3 grounded synthesis")
    parser.add_argument("--input", required=True,
                        help="Path to Stage-2 JSONL input file")
    parser.add_argument("--output", required=True,
                        help="Path to Stage-3 JSONL output file")
    parser.add_argument("--system_prompt", required=True,
                        help="Path to system prompt text file")
    parser.add_argument("--user_prompt", required=True,
                        help="Path to user prompt text file")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Model name (e.g., gpt-4o, gpt-5-chat-latest)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    asyncio.run(run_stage3_async(
        input_path=args.input,
        output_path=args.output,
        system_prompt_path=args.system_prompt,
        user_prompt_path=args.user_prompt,
        model=args.model,
        temperature=args.temperature,
        limit=args.limit,
        concurrency=args.concurrency,
    ))