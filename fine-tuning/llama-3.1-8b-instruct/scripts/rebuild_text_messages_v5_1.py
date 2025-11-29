#!/usr/bin/env python3
"""
Rebuild {train,val,test}_v5.1.text.messages_v5_final.jsonl from {split}_v5.jsonl.

- Input:  data/splits/{split}_v5.jsonl   (one JSON object per line)
- Output: data/splits/{split}_v5.1.text.messages_v5_final.jsonl

Each output line is:
  {
    "id": <id>,
    "messages": [
      {"role": "system", "content": <system_text_mode_prompt_v5.1>},
      {"role": "user", "content": <Inputs: query, retrieved_docs, per_doc_notes>},
      {"role": "assistant", "content": <STRICT TEXT-MODE answer>}
    ]
  }

The script:
  * Calls OpenAI chat
  * Normalizes the conflict label casing to match CONFLICT_TYPES
  * Ensures sentinel [[END-OF-ANSWER]] is present
  * Validates with is_format_faithful()
  * Retries a few times if the model violates the contract
"""

import os
import json
import time
from pathlib import Path
import re
from typing import Tuple, List, Dict, Any

from openai import OpenAI

# ==========================
# Config
# ==========================

MODEL_NAME = "gpt-4.1-mini"      # adjust to your deployed model
MAX_RETRIES = 3
TEMPERATURE = 0.0

BASE_DIR = Path(__file__).resolve().parents[1]  # repo root
DATA_DIR = BASE_DIR / "data" / "splits"
PROMPT_PATH = BASE_DIR / "prompts" / "sft" / "system_text_mode_sft.v5.1.txt"

OUT_PATTERN = "{split}_v5.1.text.messages_v5_final.jsonl"
IN_PATTERN = "{split}_v5.jsonl"

client = OpenAI()

# ==========================
# Label definitions
# ==========================

CONFLICT_TYPES = [
    "No conflict",
    "Complementary information",
    "Conflicting opinions or research outcomes",
    "Conflict due to outdated information",
    "Conflict due to misinformation",
]
TYPE2IDX = {t: i for i, t in enumerate(CONFLICT_TYPES)}
TYPE2IDX_NORM = {t.lower(): i for t, i in TYPE2IDX.items()}

DASH_SEPS = (" — ", " – ", " - ")

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
SENTINEL = "[[END-OF-ANSWER]]"


# ==========================
# JSONL utils
# ==========================

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for ln, s in enumerate(f, 1):
            s = s.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                raise ValueError(f"{path}:{ln} bad json: {e}")


def append_jsonl(path: Path, obj: Dict[str, Any]):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ==========================
# Format validators (same logic as train_qlora_v5)
# ==========================

def has_single_think_block(text: str) -> bool:
    opens = [m.start() for m in re.finditer(re.escape(THINK_OPEN), text)]
    closes = [m.start() for m in re.finditer(re.escape(THINK_CLOSE), text)]
    if len(opens) != 1 or len(closes) != 1:
        return False
    return opens[0] < closes[0]


def _span_after_first_top_level_json_array(text: str) -> Tuple[int, int]:
    start = text.find("[")
    if start < 0:
        return (None, None)
    depth, end, in_str, esc = 0, None, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
    return (start, end) if end is not None else (None, None)


def _find_conflict_line_span(text: str) -> Tuple[int, int]:
    # Search AFTER the first top-level JSON array
    _, array_end = _span_after_first_top_level_json_array(text)
    if array_end is None:
        return (None, None)
    tail = (text[array_end:] or "").lstrip("\n")
    offset0 = len(text) - len(tail)
    for ln in tail.splitlines():
        line = ln.strip()
        if not line:
            offset0 += len(ln) + 1
            continue
        for sep in DASH_SEPS:
            if sep in line:
                left = line.split(sep, 1)[0].strip()
                # CASE-INSENSITIVE label detection
                if left.lower() in TYPE2IDX_NORM:
                    s = text.find(line, offset0)
                    if s >= 0:
                        return (s, s + len(line))
        offset0 += len(ln) + 1
    return (None, None)


def extract_conflict_label(text: str):
    ls, le = _find_conflict_line_span(text)
    if ls is None:
        return None
    line = text[ls:le]
    for sep in DASH_SEPS:
        if sep in line:
            left = line.split(sep, 1)[0].strip()
            return TYPE2IDX_NORM.get(left.lower(), None)
    return None


def is_format_faithful(assistant: str) -> bool:
    if not has_single_think_block(assistant):
        return False
    if SENTINEL not in assistant:
        return False
    if extract_conflict_label(assistant) is None:
        return False
    return True


# ==========================
# Post-processing helpers
# ==========================

def normalize_conflict_label_casing(assistant: str) -> str:
    """
    Force the ConflictType to use canonical casing from CONFLICT_TYPES.
    """
    ls, le = _find_conflict_line_span(assistant)
    if ls is None:
        return assistant

    line = assistant[ls:le]
    for sep in DASH_SEPS:
        if sep in line:
            left, right = line.split(sep, 1)
            left_clean = left.strip()
            idx = TYPE2IDX_NORM.get(left_clean.lower())
            if idx is None:
                return assistant
            canonical = CONFLICT_TYPES[idx]
            new_line = f"{canonical}{sep}{right.lstrip()}"
            if new_line != line:
                return assistant[:ls] + new_line + assistant[le:]
            else:
                return assistant
    return assistant


def ensure_sentinel(assistant: str) -> str:
    if SENTINEL in assistant:
        return assistant
    # Append sentinel on a new line
    if not assistant.endswith("\n"):
        assistant = assistant + "\n"
    return assistant + SENTINEL


def postprocess_assistant(text: str) -> str:
    text = normalize_conflict_label_casing(text)
    text = ensure_sentinel(text)
    return text


# ==========================
# Prompt building
# ==========================

def load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def format_docs(docs: List[Dict[str, Any]]) -> str:
    # Pretty-print JSON so the model can see structure clearly
    return json.dumps(docs, ensure_ascii=False, indent=2)


def format_per_doc_notes(notes: List[Dict[str, Any]]) -> str:
    return json.dumps(notes, ensure_ascii=False, indent=2)


def build_user_message(ex: Dict[str, Any]) -> str:
    """
    Build the 'user' content, closely mirroring the example you showed.
    Assumes ex has: id, query, retrieved_docs, per_doc_notes
    """
    query = ex.get("query", "")
    retrieved_docs = ex.get("retrieved_docs", [])
    per_doc_notes = ex.get("per_doc_notes", [])

    return f"""Inputs:

- query:
{query}

- retrieved_docs (ordered d1…dN):
{format_docs(retrieved_docs)}

- per_doc_notes (for each doc_id; includes verdict, key_fact, verdict_reason, source_quality):
{format_per_doc_notes(per_doc_notes)}

Task:
1) Follow the full OUTPUT CONTRACT exactly.
   • <think> block with:
       (A) VALID JSON array for EVERY doc d1…dN (order-preserving, one object per doc; if verdict=="irrelevant" set key_fact="")
       (B) Conflict reasoning FIRST: cluster docs, reference doc IDs, and NAME the mechanism (temporal / factual-accuracy / contextual-scope / methodological / linguistic-interpretive)
       (C) ONE label line: "<ConflictType> — <concise conflict_reason>"
       (D) Brief reasoning connecting evidence to the final answer (or abstention)
   • ONE blank line
   • Final answer (or exactly "CANNOT ANSWER, INSUFFICIENT EVIDENCE" if abstaining)
   • Final sentinel line [[END-OF-ANSWER]].

Reminders (DO NOT PRINT):
- Conflict taxonomy (strict): No conflict / Complementary information / Conflicting opinions or research outcomes / Conflict due to outdated information / Conflict due to misinformation.
- Use only existing doc_ids in bracketed citations [dX]; no ranges like d1–d5; never cite out-of-bounds [dK].
- Prefer high-credibility sources and order citations high→low in the evidence list.
- If any doc is "supports" or "partially supports", DO NOT abstain.
- Close </think> before the answer; no extra text outside the required format.
"""


# ==========================
# OpenAI call
# ==========================

def call_model(system_prompt: str, user_content: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return resp.choices[0].message.content


# ==========================
# Main rebuild logic
# ==========================

def rebuild_split(split: str):
    in_path = DATA_DIR / IN_PATTERN.format(split=split)
    out_path = DATA_DIR / OUT_PATTERN.format(split=split)

    if not in_path.is_file():
        print(f"[{split}] Input not found: {in_path}")
        return

    # Simple resume: collect existing IDs if output already exists
    done_ids = set()
    if out_path.is_file():
        print(f"[{split}] Output exists, resuming from: {out_path}")
        for ex in read_jsonl(out_path):
            ex_id = ex.get("id")
            if ex_id is not None:
                done_ids.add(ex_id)
        print(f"[{split}] Found {len(done_ids)} already completed examples")

    system_prompt = load_system_prompt()

    total = 0
    good = 0
    bad = 0

    for ex in read_jsonl(in_path):
        ex_id = ex.get("id")
        total += 1

        if ex_id in done_ids:
            continue

        user_content = build_user_message(ex)

        assistant = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = call_model(system_prompt, user_content)
                fixed = postprocess_assistant(raw)
                if is_format_faithful(fixed):
                    assistant = fixed
                    break
                else:
                    print(f"[{split}] id={ex_id} attempt={attempt} FAILED format check, retrying...")
            except Exception as e:
                print(f"[{split}] id={ex_id} attempt={attempt} error: {e}")
                time.sleep(2 * attempt)

        if assistant is None:
            # Last resort: keep the last fixed text even if not perfect
            print(f"[{split}] id={ex_id} could not get faithful format after {MAX_RETRIES} attempts; keeping last attempt with best post-processing.")
            # Try one more call but don't loop forever
            try:
                raw = call_model(system_prompt, user_content)
                assistant = postprocess_assistant(raw)
            except Exception as e:
                print(f"[{split}] id={ex_id} final call error: {e}")
                assistant = f"<think>[]\nNo conflict — fallback\nUsing fallback due to generation errors.\n</think>\n\nCANNOT ANSWER, INSUFFICIENT EVIDENCE\n{SENTINEL}"
            bad += 1
        else:
            good += 1

        out_obj = {
            "id": ex_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant},
            ],
        }
        append_jsonl(out_path, out_obj)

        if (good + bad) % 50 == 0:
            print(f"[{split}] progress: total_seen={total} newly_written={good+bad} (good={good}, bad={bad})")

    print(f"[{split}] DONE. Newly written: {good+bad}, good={good}, bad={bad}. Output → {out_path}")


def main():
    for split in ["train", "val", "test"]:
        rebuild_split(split)


if __name__ == "__main__":
    main()