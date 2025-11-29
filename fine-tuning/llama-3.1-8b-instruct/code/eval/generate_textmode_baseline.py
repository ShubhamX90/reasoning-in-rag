#!/usr/bin/env python3
"""
TEXT-MODE baseline generator (v5, end-to-end, messages_v5_final)
----------------------------------------------------------------
- Uses *.text.messages_v5_final.jsonl-style inputs:
    {"id": "...",
     "messages": [
        {"role":"system","content":"..."},
        {"role":"user","content":"..."},
        {"role":"assistant","content":"..."}  # gold, ignored
     ]
    }

- For generation, it:
    • Overrides the system message with the v5 system prompt file.
    • Reuses the user message text exactly as stored.
    • Ignores the gold assistant message.

- No LoRA: pure BASE model baseline.
- Uses sentinel [[END-OF-ANSWER]] to stop cleanly.
- Optional 4-bit inference via --load_in_4bit (or env GEN_INT4=1).
- Safe left-side truncation so prompt+generation never exceed context.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

SENTINEL = "[[END-OF-ANSWER]]"


# ---------------- IO ----------------
def read_jsonl(p: str):
    with open(p, "r", encoding="utf-8") as f:
        for ln, s in enumerate(f, 1):
            s = s.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                raise ValueError(f"{p}:{ln} bad json: {e}")


def load_text(p: str) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


# ---------------- Prompt helpers ----------------
def _extract_user_content(msgs: List[Dict[str, str]]) -> str:
    """
    Given messages from *.text.messages_v5_final.jsonl, return the user text.
    Expect roles ["system", "user", "assistant"], but we only rely on role.
    """
    for m in msgs:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def build_messages_from_final(
    system_txt: str,
    ex: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Build chat messages for baseline generation from a final-messages example:
      {"id": "...", "messages": [...]}

    - Override system with v5 system prompt.
    - Keep user content as stored.
    - Ignore the gold assistant message.
    """
    msgs = ex.get("messages", [])
    if not isinstance(msgs, list) or not msgs:
        user_txt = ""
    else:
        user_txt = _extract_user_content(msgs)

    return [
        {"role": "system", "content": system_txt},
        {"role": "user", "content": user_txt},
    ]


def estimate_doc_count_from_user(user_txt: str) -> int:
    """
    Heuristic: count occurrences of `"doc_id":` in the user message,
    which is how {retrieved_docs} is rendered in the v5 user templates.
    """
    return max(0, user_txt.count('"doc_id"'))


# ---------------- Stopping on sentinel ----------------
class SentinelStopper(StoppingCriteria):
    def __init__(self, tokenizer: AutoTokenizer, sentinel: str):
        super().__init__()
        self.sentinel_ids = tokenizer.encode(sentinel, add_special_tokens=False)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:
        if input_ids.shape[0] == 0 or len(self.sentinel_ids) == 0:
            return False
        seq = input_ids[0].tolist()
        n = len(self.sentinel_ids)
        return len(seq) >= n and seq[-n:] == self.sentinel_ids


# ---------------- Length heuristic ----------------
def estimate_max_new_tokens(n_docs: int, base: int, cap: int) -> int:
    """
    Heuristic: ~70 tokens per doc JSON object + 240 for conflict+reasoning+final.
    Scales with doc count but obeys a hard cap; adds generous floors for many docs.
    """
    est = int(240 + 70 * max(1, n_docs))
    if n_docs >= 12:
        est = max(est, 1600)
    if n_docs >= 16:
        est = max(est, 2000)
    return max(base, min(est, cap))


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument(
        "--input_jsonl",
        required=True,
        help="*.text.messages_v5_final.jsonl (e.g., data/splits/test_v5.text.messages_v5_final.jsonl)",
    )
    ap.add_argument(
        "--system_prompt",
        required=True,
        help="v5 system prompt (e.g., prompts/sft/system_text_mode_sft.v5.txt)",
    )
    ap.add_argument("--out_jsonl", required=True)

    ap.add_argument("--attn_impl", choices=["eager", "sdpa"], default="sdpa")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)

    ap.add_argument(
        "--auto_length",
        action="store_true",
        help="Adjust max_new_tokens per example based on doc count in user message",
    )
    ap.add_argument("--max_new_tokens_base", type=int, default=1200)
    ap.add_argument("--max_new_tokens_cap", type=int, default=2200)

    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of examples (0 = all)",
    )

    ap.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Quantize base model to 4-bit for generation (or set GEN_INT4=1)",
    )
    ap.add_argument(
        "--local_files_only",
        action="store_true",
        help="Force offline HF cache usage",
    )

    args = ap.parse_args()

    # ---------------- Preflight: sanity on input schema ----------------
    items_all = list(read_jsonl(args.input_jsonl))
    if args.limit > 0:
        items = items_all[: args.limit]
    else:
        items = items_all

    with_msgs = sum(
        1
        for x in items
        if isinstance(x.get("messages"), list)
        and any(m.get("role") == "user" for m in x["messages"])
    )
    print(
        f"[check] examples={len(items)}, with_user_messages={with_msgs}, "
        f"missing_user_messages={len(items) - with_msgs}"
    )
    if with_msgs == 0:
        print(
            "[FATAL] input_jsonl has no usable 'messages' with a user role. "
            "Use *.text.messages_v5_final.jsonl built with the v5 builder."
        )
        out_path = Path(args.out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as wf:
            pass
        return

    # ---------------- Tokenizer ----------------
    tok = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # Optional int4 quantization
    use_int4 = args.load_in_4bit or (os.environ.get("GEN_INT4", "0") == "1")
    quant_cfg = None
    if use_int4:
        from transformers import BitsAndBytesConfig

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # ---------------- Base model (no LoRA) ----------------
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=args.attn_impl,
        quantization_config=quant_cfg,
        local_files_only=args.local_files_only,
    )
    model.eval()

    # Context management / safe truncation
    max_ctx = getattr(model.config, "max_position_embeddings", 8192)
    tok.model_max_length = max_ctx
    tok.truncation_side = "left"
    safety = 32  # tiny buffer

    # ---------------- System prompt ----------------
    sys_txt = load_text(args.system_prompt)

    # ---------------- Generation loop ----------------
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stopper = StoppingCriteriaList([SentinelStopper(tok, SENTINEL)])

    with open(out_path, "w", encoding="utf-8") as wf:
        for ex in items:
            cid = ex.get("id")
            msgs = ex.get("messages", [])
            user_txt = _extract_user_content(msgs) if isinstance(msgs, list) else ""

            chat_msgs = build_messages_from_final(sys_txt, ex)

            prompt = tok.apply_chat_template(
                chat_msgs,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Heuristic: estimate doc count from user message
            n_docs = estimate_doc_count_from_user(user_txt)
            if args.auto_length:
                max_new = estimate_max_new_tokens(
                    n_docs,
                    args.max_new_tokens_base,
                    args.max_new_tokens_cap,
                )
            else:
                max_new = args.max_new_tokens_base

            # safe input length = max_ctx - max_new - safety
            max_inp = max(512, int(max_ctx) - int(max_new) - safety)
            inputs = tok(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_inp,
            ).to(model.device)

            gen = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=(args.temperature > 0.0),
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
                stopping_criteria=stopper,
            )
            out_text = tok.decode(
                gen[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            )

            wf.write(
                json.dumps({"id": cid, "raw": out_text}, ensure_ascii=False) + "\n"
            )

    print(f"[BASELINE v5] Wrote → {out_path}")


if __name__ == "__main__":
    main()