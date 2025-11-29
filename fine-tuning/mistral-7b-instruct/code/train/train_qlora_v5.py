#!/usr/bin/env python3
# v5.3.2-final-only (with v5.1 prompts)
# - Accepts ONLY *.text.messages_v5_final.jsonl (train/dev)
# - Reads prebuilt messages; no raw/messages autodetection
# - Overrides system with v5.1 template; validates assistant TEXT-MODE
# - Weighted loss on conflict-label line; QLoRA; macro-F1 early stop
# - DevMacroF1 uses extract_conflict_label on generated TEXT-MODE
# - Softer class balancing (1/sqrt(count))

import os, json, math, argparse, re, random
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, BitsAndBytesConfig, TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------- labels & patterns ----------------
CONFLICT_TYPES = [
    "No conflict",
    "Complementary information",
    "Conflicting opinions or research outcomes",
    "Conflict due to outdated information",
    "Conflict due to misinformation",
]
TYPE2IDX = {t: i for i, t in enumerate(CONFLICT_TYPES)}
IDX2TYPE = {i: t for t, i in TYPE2IDX.items()}
DASH_SEPS = (" — ", " – ", " - ")

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
SENTINEL = "[[END-OF-ANSWER]]"

# ---------------- jsonl utils ----------------
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for ln, s in enumerate(f, 1):
            s = s.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                raise ValueError(f"{path}:{ln} bad json: {e}")

# ---------------- formatting validators ----------------
def has_single_think_block(text: str):
    opens = [m.start() for m in re.finditer(re.escape(THINK_OPEN), text)]
    closes = [m.start() for m in re.finditer(re.escape(THINK_CLOSE), text)]
    if len(opens) != 1 or len(closes) != 1:
        return False
    return opens[0] < closes[0]

def _span_after_first_top_level_json_array(text: str):
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

def _find_conflict_line_span(text: str):
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
                if left in TYPE2IDX:
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
            return TYPE2IDX.get(left, None)
    return None

def is_format_faithful(assistant: str):
    if not has_single_think_block(assistant):
        return False
    if SENTINEL not in assistant:
        return False
    if extract_conflict_label(assistant) is None:
        return False
    return True

# ---------------- load prompts ----------------
def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

# ---------------- final-only guard ----------------
def _assert_final_jsonl_pathlike(p: str, arg_name: str):
    if not p.endswith(".text.messages_v5_final.jsonl"):
        raise ValueError(
            f"{arg_name} must point to a '*.text.messages_v5_final.jsonl' file. Got: {p}"
        )
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{arg_name} not found: {p}")

# ---------------- dataset ----------------
class ChatTextModeSFTV5FinalOnly(Dataset):
    """
    FINAL-ONLY dataset:
      • Accepts ONLY prebuilt *.text.messages_v5_final.jsonl
      • Overrides system with v5.1 content
      • Validates assistant TEXT-MODE (think, label line, sentinel)
      • Supervises assistant tokens only; up-weights conflict label line
    """
    def __init__(
        self, tok, jsonl_path, system_prompt,
        max_len=8192,
        conflict_weight=2.0,
        drop_on_truncation=True,
    ):
        _assert_final_jsonl_pathlike(str(jsonl_path), "jsonl_path")
        self.tok = tok
        self.max_len = int(max_len)
        self.conf_w = float(conflict_weight)
        self.items, self.class_idx = [], []
        n_kept = n_drop = 0
        sup_total = 0

        for ex in read_jsonl(jsonl_path):
            msgs = ex.get("messages")
            if not (isinstance(msgs, list) and len(msgs) == 3):
                n_drop += 1
                continue

            # Enforce roles & override system with v5.1 prompt
            sys_m, user_m, asst_m = msgs
            if sys_m.get("role") != "system" or user_m.get("role") != "user" or asst_m.get("role") != "assistant":
                n_drop += 1
                continue

            user_and_sys = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_m.get("content", "")},
            ]
            assistant = asst_m.get("content", "")

            # Validate strict TEXT-MODE
            if not is_format_faithful(assistant):
                n_drop += 1
                continue

            # ---------------- token budgeting (preserve assistant) ----------------
            a_tok = self.tok(
                assistant, add_special_tokens=True, truncation=False, return_offsets_mapping=True
            )
            a_ids = a_tok["input_ids"]

            prompt_text = self.tok.apply_chat_template(
                user_and_sys, tokenize=False, add_generation_prompt=True
            )
            p_ids = self.tok(prompt_text, add_special_tokens=True, truncation=False)["input_ids"]

            max_len = self.max_len
            if len(p_ids) + len(a_ids) > max_len:
                budget = max_len - len(a_ids)
                if budget <= 64:
                    if drop_on_truncation:
                        n_drop += 1
                        continue
                    else:
                        a_ids = a_ids[: max(16, max_len - 64)]
                        budget = max_len - len(a_ids)
                p_ids = p_ids[: max(64, budget)]

            full_ids = p_ids + a_ids

            # SAFETY: final clamp in the unlikely case we still exceed by 1 due to special tokens
            if len(full_ids) > self.max_len:
                if drop_on_truncation:
                    n_drop += 1
                    continue
                else:
                    full_ids = full_ids[: self.max_len]

            attn = [1] * len(full_ids)

            # Labels: supervise assistant only
            plen = len(p_ids)
            labels = [-100] * len(full_ids)
            for i in range(plen, len(full_ids)):
                labels[i] = full_ids[i]
            sup_total += sum(1 for t in labels if t != -100)

            # Up-weight conflict label line tokens
            line_s, line_e = _find_conflict_line_span(assistant)
            weights = [1.0] * len(full_ids)
            if line_s is not None and line_e is not None and "offset_mapping" in a_tok:
                base = plen
                for j, (b_off, e_off) in enumerate(a_tok["offset_mapping"]):
                    if b_off is None or e_off is None:
                        continue
                    pos = base + j
                    if pos >= len(full_ids):
                        break
                    if labels[pos] != -100 and (b_off >= line_s) and (e_off <= line_e):
                        weights[pos] = self.conf_w

            lab = extract_conflict_label(assistant)
            self.class_idx.append(-1 if lab is None else lab)

            self.items.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "loss_weights": torch.tensor(weights, dtype=torch.float32),
            })
            n_kept += 1

        print(f"[Data] {jsonl_path} (final-only) → kept {n_kept}, dropped {n_drop}")
        print(f"[Data] supervised tokens total = {sup_total}")
        # Label distribution (useful for sampler sanity)
        cnt = Counter([c for c in self.class_idx if c is not None and c >= 0])
        if cnt:
            print(f"[Data] label distribution: {dict(sorted(cnt.items()))}")
        self._ensure_has_supervision()

    def _ensure_has_supervision(self):
        if not self.items:
            raise RuntimeError("Split has 0 items after normalization/validation.")
        any_tok = any((t != -100) for ex in self.items for t in ex["labels"].tolist())
        if not any_tok:
            raise RuntimeError("Split has 0 supervised tokens after truncation.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

class PadCollator:
    def __init__(self, tok):
        self.tok = tok
        self.tok.padding_side = "right"

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        def pad(v, fill):
            return torch.cat([v, torch.full((max_len - len(v),), fill, dtype=v.dtype)], dim=0)
        input_ids = torch.stack([pad(x["input_ids"], self.tok.pad_token_id) for x in batch])
        attention_mask = torch.stack([pad(x["attention_mask"], 0) for x in batch])
        labels = torch.stack([pad(x["labels"], -100) for x in batch])
        weights = torch.stack([pad(x["loss_weights"], 1.0) for x in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_weights": weights,
        }

# ---------------- trainer with weighted CE ----------------
class WeightedTokenTrainer(Trainer):
    # Accept & ignore extra kwargs from newer HF (e.g., num_items_in_batch)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        weights = inputs.pop("loss_weights")
        outputs = model(**inputs, labels=labels)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = weights[..., 1:].contiguous()

        ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.size())

        active = (shift_labels != -100).float()
        mult = active * shift_weights
        denom = mult.sum().clamp_min(1.0)
        loss = (ce * mult).sum() / denom
        return (loss, outputs) if return_outputs else loss

# ---------------- quick dev callback (macro-F1 on labels) ----------------
class DevMacroF1Callback(TrainerCallback):
    def __init__(self, tok, dev_path, out_dir, system_prompt, patience=4, max_new_tokens=160):
        _assert_final_jsonl_pathlike(dev_path, "dev_jsonl")
        self.tok = tok
        self.dev = list(read_jsonl(dev_path))
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.best = -1.0
        self.bad = 0
        self.patience = int(patience)
        self.max_new = int(max_new_tokens)
        self.system_prompt = system_prompt

        # Cache user text + gold label from assistant in messages
        self.cache = []
        for ex in self.dev:
            msgs = ex.get("messages") or []
            user_txt = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
            gold = extract_conflict_label((msgs[-1] or {}).get("content", "")) if msgs else None
            self.cache.append((user_txt, gold))

    def _quick_label(self, model, user_text):
        # Build a fresh system+user chat, use the same system prompt as training
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
        ]
        prompt = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

        max_ctx = int(getattr(self.tok, "model_max_length", 8192))
        safety = 64
        max_inp = max(384, max_ctx - self.max_new - safety)

        inputs = self.tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_inp,
        ).to(model.device)

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=self.max_new,
                do_sample=False,
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.pad_token_id,
            )

        # Only the continuation beyond the prompt
        cont = gen[0][inputs["input_ids"].shape[-1]:]
        out = self.tok.decode(cont, skip_special_tokens=True)

        # Use the same parser as in training/eval to extract the conflict label
        return extract_conflict_label(out)

    @staticmethod
    def _macro_f1(golds, preds, n=5):
        cm = [[0] * n for _ in range(n)]
        for g, p in zip(golds, preds):
            if g is None or p is None:
                continue
            cm[g][p] += 1
        f1s = []
        for c in range(n):
            tp = cm[c][c]
            fp = sum(cm[r][c] for r in range(n)) - tp
            fn = sum(cm[c][r] for r in range(n)) - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
        return sum(f1s) / n

    def on_epoch_end(self, args, state, control, **kw):
        model = kw["model"].eval()
        preds, golds = [], []
        for user_txt, g in self.cache:
            p = self._quick_label(model, user_txt)
            preds.append(p)
            golds.append(g)
        macro = self._macro_f1(golds, preds)
        print(f"[DevMacroF1] epoch={getattr(state, 'epoch', -1):.2f} macro-F1={macro:.4f}")
        if macro > self.best + 1e-6:
            self.best = macro
            self.bad = 0
            best_dir = self.out_dir / "best_dev_f1"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir.as_posix())
            print(f"[DevMacroF1] new BEST saved → {best_dir}")
        else:
            self.bad += 1
            if self.bad >= self.patience:
                print(f"[DevMacroF1] early stopping (patience={self.patience})")
                control.should_training_stop = True
        return control

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--train_jsonl", required=True)  # MUST be *.text.messages_v5_final.jsonl
    ap.add_argument("--dev_jsonl", required=True)    # MUST be *.text.messages_v5_final.jsonl
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--system_prompt_path", required=True)  # prompts/sft/system_text_mode_sft.v5.1.txt

    # Common knobs (kept to match your launch script)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--attn_impl", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--conflict_weight", type=float, default=2.0)
    ap.add_argument("--dev_patience", type=int, default=4)
    ap.add_argument("--dev_max_new", type=int, default=160)
    ap.add_argument("--resume_from", type=str, default=None)

    args = ap.parse_args()

    # Guards: final-only paths
    _assert_final_jsonl_pathlike(args.train_jsonl, "train_jsonl")
    _assert_final_jsonl_pathlike(args.dev_jsonl, "dev_jsonl")

    # Stability/VRAM hygiene
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(42)
    random.seed(42)

    # Load v5.1 system prompt
    system_prompt = load_text(args.system_prompt_path)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.model_max_length = args.max_len

    # QLoRA base
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
        local_files_only=True,
    )

    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    lconf = LoraConfig(
        r=32, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM", bias="none",
    )
    model = get_peft_model(base, lconf)
    model.print_trainable_parameters()

    # Data
    train_ds = ChatTextModeSFTV5FinalOnly(
        tok, args.train_jsonl, system_prompt,
        max_len=args.max_len,
        conflict_weight=args.conflict_weight,
        drop_on_truncation=True,
    )
    dev_ds = ChatTextModeSFTV5FinalOnly(
        tok, args.dev_jsonl, system_prompt,
        max_len=args.max_len,
        conflict_weight=args.conflict_weight,
        drop_on_truncation=True,
    )
    collator = PadCollator(tok)

    # Class-balanced sampler on conflict labels (softer: 1/sqrt(count))
    counts = Counter([c for c in train_ds.class_idx if c is not None and c >= 0])
    if counts:
        per_ex_w = []
        for c in train_ds.class_idx:
            cnt = max(1, counts.get(c, 1))
            per_ex_w.append(1.0 / (cnt ** 0.5))
        sampler = WeightedRandomSampler(
            torch.as_tensor(per_ex_w, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader_kwargs = dict(sampler=sampler)
    else:
        train_loader_kwargs = dict(shuffle=True)

    steps_per_epoch = max(1, math.ceil(len(train_ds) / (args.bsz * args.grad_accum)))

    targs = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=max(1, steps_per_epoch // 10),
        eval_strategy="no",
        save_strategy="steps",
        save_steps=max(1, steps_per_epoch // 2),
        save_total_limit=2,
        bf16=True,
        report_to="none",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        group_by_length=True,
        dataloader_num_workers=2,
        save_safetensors=True,
        seed=42,
        overwrite_output_dir=True,
    )

    class WeightedTrainer(WeightedTokenTrainer):
        pass

    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
    )

    # plug our dataloader to honor the custom sampler
    def _get_train_dl():
        return DataLoader(
            train_ds,
            batch_size=targs.per_device_train_batch_size,
            collate_fn=collator,
            pin_memory=True,
            **train_loader_kwargs,
        )
    trainer.get_train_dataloader = _get_train_dl

    # Lightweight dev macro-F1 early stop (final-only)
    trainer.add_callback(
        DevMacroF1Callback(
            tok,
            args.dev_jsonl,
            args.out_dir,
            system_prompt=system_prompt,
            patience=args.dev_patience,
            max_new_tokens=args.dev_max_new,
        )
    )

    trainer.train(resume_from_checkpoint=args.resume_from)
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print("Saved adapter →", args.out_dir)

if __name__ == "__main__":
    main()