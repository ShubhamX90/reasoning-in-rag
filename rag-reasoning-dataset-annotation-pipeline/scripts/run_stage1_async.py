#!/usr/bin/env python3
"""
Stage-1 Async: Concurrent Micro Evidence Adjudication
-----------------------------------------------------
Modified to support multiline JSON input files (non-JSONL).
Each query block may span multiple lines.

usage:

python3 -m scripts.run_stage1_async1 \
  --input data/normalized/refusals11.jsonl \                                             
  --output data/stage1_outputs/refusals11_stage1.jsonl \                                  
  --concurrency 12                           


"""

import os, json, argparse, asyncio, re
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

SYSTEM_PROMPT_PATH = "prompts/system_stage1.txt"
USER_PROMPT_TEMPLATE_PATH = "prompts/user_stage1.txt"
MODEL_NAME = "gpt-5-chat-latest"
MAX_RETRIES = 3

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

system_prompt = load_text(SYSTEM_PROMPT_PATH)
user_template = load_text(USER_PROMPT_TEMPLATE_PATH)

def _load_api_key() -> str:
    key_path = os.path.expanduser("~/.openai_key")
    if os.path.exists(key_path):
        return open(key_path).read().strip()
    if "OPENAI_API_KEY" in os.environ:
        return os.environ["OPENAI_API_KEY"].strip()
    raise RuntimeError("❌ No API key found. Set OPENAI_API_KEY or ~/.openai_key")

def _brace_safe_fill(template: str, mapping: dict) -> str:
    temp = template
    for k in mapping.keys():
        temp = temp.replace("{" + k + "}", f"@@{k}@@")
    temp = temp.replace("{", "{{").replace("}", "}}")
    for k, v in mapping.items():
        temp = temp.replace(f"@@{k}@@", "" if v is None else str(v))
    return temp

def _extract_json(text: str):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        return json.loads(match.group(0))
    except Exception:
        return None

def _valid_note(note: dict) -> bool:
    required = ["doc_id", "verdict", "key_fact", "quote", "verdict_reason", "source_quality"]
    allowed_verdicts = {"supports", "partially supports", "irrelevant"}
    allowed_quality = {"high", "low"}
    for k in required:
        if k not in note:
            return False
    if note["verdict"] not in allowed_verdicts or note["source_quality"] not in allowed_quality:
        return False
    if note["verdict"] == "irrelevant":
        return note["key_fact"] == "" and note["quote"] == ""
    return bool(note["key_fact"]) and bool(note["quote"])

# -------- Load flexible JSON input --------
def load_records_flex(path: str):
    text = Path(path).read_text(encoding="utf-8")
    text = re.sub(r"(\})(\s*\{)", r"\1\n\2", text.strip())  # split JSON blocks
    blocks = [b.strip() for b in text.split("\n") if b.strip()]
    records = []
    for block in blocks:
        try:
            fixed = re.sub(r"}\s*{", "}, {", block)
            record = json.loads(fixed)
            records.append(record)
        except Exception as e:
            print(f"⚠️ Skipping invalid block: {e}")
    return records

# ---------- Async LLM call ----------
async def adjudicate_doc(client, semaphore, query_text, record_id, doc):
    async with semaphore:
        doc_id = doc.get("doc_id", "")
        snippet = doc.get("snippet", "")
        url = doc.get("source_url", "")
        timestamp = doc.get("timestamp", "") or ""

        user_prompt = _brace_safe_fill(
            user_template,
            {"QUERY": query_text, "DOC_ID": doc_id, "URL": url, "TEXT": snippet, "TIMESTAMP": timestamp},
        )

        backoff = 1.0
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL_NAME,
                    temperature=0.2,
                    max_tokens=350,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    extra_headers={"Idempotency-Key": f"{record_id}:{doc_id}"},
                    timeout=60.0,
                    response_format={"type": "json_object"},
                )
                content = (resp.choices[0].message.content or "").strip()
                parsed = _extract_json(content)
                if parsed and _valid_note(parsed):
                    return parsed
            except Exception as e:
                err = str(e)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10.0)

        return {
            "doc_id": doc_id,
            "verdict": "irrelevant",
            "key_fact": "",
            "quote": "",
            "verdict_reason": "Fallback or parsing failure after retries.",
            "source_quality": "low",
        }

# ---------- Main async ----------
async def run_async_stage1(input_path: str, output_path: str, limit: int | None, concurrency: int):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(api_key=_load_api_key())
    semaphore = asyncio.Semaphore(concurrency)
    out_lock = asyncio.Lock()

    processed = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    processed.add(json.loads(line).get("id"))
                except Exception:
                    pass
        if processed:
            print(f"⏩ Resuming from {len(processed)} processed queries")

    records = load_records_flex(input_path)
    records = [r for r in records if r.get("id") not in processed]
    if limit:
        records = records[:limit]
    total = len(records)
    print(f"⚙️ Processing {total} new queries (concurrency={concurrency})")

    async def process_record(idx, record):
        record_id = record.get("id") or f"#{idx+1:04d}"
        query = record.get("query", "")
        docs = record.get("retrieved_docs", [])

        per_doc_notes = await asyncio.gather(
            *[adjudicate_doc(client, semaphore, query, record_id, d) for d in docs]
        )
        record["per_doc_notes"] = per_doc_notes

        async with out_lock:
            with open(output_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    await tqdm_asyncio.gather(
        *[process_record(i, rec) for i, rec in enumerate(records)],
        total=total,
        desc="Stage-1 async progress",
    )

    print(f"✅ Stage-1 completed. Results written to {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run concurrent Stage-1 micro adjudication.")
    ap.add_argument("--input", required=True, help="Input file (multiline JSON)")
    ap.add_argument("--output", required=True, help="Output JSONL")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()
    asyncio.run(run_async_stage1(args.input, args.output, args.limit, args.concurrency))