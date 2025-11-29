#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate {train, val, test}.text.messages_v5_final.jsonl against v5 system/user prompts.

Checks:
  • JSONL well-formed; each row has messages=[{system},{user},{assistant}] in order.
  • System message content EXACTLY matches the current system template file
    (normalized newlines, trimmed trailing whitespace).
  • User message adheres to the v5 scaffold:
      - has the three "Inputs" sections with anchors in the right order,
      - we extract retrieved_docs & per_doc_notes JSON and parse them,
      - the "Task:" block (from 'Task:\n' to end) must exactly match the v5 user template suffix.
  • Assistant message (TEXT-MODE output contract):
      - exactly one "<think>" and one "</think>", opening before closing,
      - the literal string "<think>" does NOT appear inside the block,
      - exactly ONE blank line between </think> and final answer,
      - think(A): first element is a VALID JSON array; doc_ids are exactly d1..dN,
        verdict ∈ {"supports","partially supports","irrelevant"}, source_quality ∈ {"high","low"},
        if verdict=="irrelevant" ⇒ key_fact=="", else key_fact word-count ≤ 80,
        verdict_reason word-count ≤ 80,
        len(array) == len(retrieved_docs) parsed from user prompt.
      - think(B): contains at least one doc-id reference and NAMES a mechanism
        in {"temporal","factual-accuracy","contextual-scope","methodological","linguistic-interpretive"}.
      - think(C): has a single label line using EM DASH " — " and a known conflict type with reason ≤ 50 words.
      - After </think>:
          • Either the exact abstain line "CANNOT ANSWER, INSUFFICIENT EVIDENCE"
          • OR 2–4 sentences total with ≥1 bracketed citation like [dK] and no OOB citations (K∈[1..N]).
      - Must end with sentinel '[[END-OF-ANSWER]]'.

Usage:
  python code/data/validate_text_messages_v5.py \
    --splits_dir data/splits \
    --system_file prompts/sft/system_text_mode_sft.v5.txt \
    --user_file   prompts/sft/user_text_mode_sft.v5.txt
"""

import argparse, json, os, re, sys
from collections import Counter, defaultdict

ALLOWED_VERDICTS = {"supports", "partially supports", "irrelevant"}
ALLOWED_QUALITY = {"high", "low"}
MECHANISMS = {"temporal", "factual-accuracy", "contextual-scope", "methodological", "linguistic-interpretive"}
CONFLICT_TYPES = {
    "No conflict",
    "Complementary information",
    "Conflicting opinions or research outcomes",
    "Conflict due to outdated information",
    "Conflict due to misinformation"
}

def norm(s: str) -> str:
    # Normalize newlines and strip trailing whitespace at ends
    return re.sub(r'\r\n?', '\n', s).strip()

def word_count(s: str) -> int:
    return len(re.findall(r"\w+", s))

def load_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_user_template_suffix(user_tpl: str) -> str:
    # From "Task:\n" till end
    m = re.search(r"\n\nTask:\n", user_tpl)
    if not m:
        raise RuntimeError("Could not locate 'Task:' section in user template.")
    return user_tpl[m.start()+2:]  # keep the leading "\n\nTask:\n" for exact match

def split_user_sections(user_txt: str):
    """
    Return (query_str, retrieved_docs_json_text, per_doc_notes_json_text, task_suffix_text)
    by carving the user message using the fixed anchors.
    """
    anchors = [
        "Inputs:\n\n- query:\n",
        "\n\n- retrieved_docs (ordered d1…dN):\n",
        "\n\n- per_doc_notes (for each doc_id; includes verdict, key_fact, verdict_reason, source_quality):\n",
        "\n\nTask:\n"
    ]
    for a in anchors:
        if a not in user_txt:
            raise ValueError(f"User message missing anchor: {a!r}")

    # positions
    i0 = user_txt.index(anchors[0]) + len(anchors[0])
    i1 = user_txt.index(anchors[1])
    i2 = user_txt.index(anchors[1]) + len(anchors[1])
    i3 = user_txt.index(anchors[2])
    i4 = user_txt.index(anchors[2]) + len(anchors[2])
    i5 = user_txt.index(anchors[3])

    query_str = user_txt[i0:i1]
    retrieved_docs_txt = user_txt[i2:i3]
    per_doc_notes_txt = user_txt[i4:i5]
    task_suffix = user_txt[i5:]  # includes "\n\nTask:\n..."

    return query_str, retrieved_docs_txt, per_doc_notes_txt, task_suffix

def parse_json_block(txt: str, field_name: str):
    try:
        return json.loads(txt)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON for {field_name}: {e}\nSnippet start: {txt[:200]!r}")

def find_think_block(assistant: str):
    if assistant.count("<think>") != 1:
        raise ValueError(f"Assistant must contain exactly one '<think>' (found {assistant.count('<think>')}).")
    if assistant.count("</think>") != 1:
        raise ValueError(f"Assistant must contain exactly one '</think>' (found {assistant.count('</think>')}).")
    start = assistant.index("<think>") + len("<think>")
    end = assistant.index("</think>")
    if start > end:
        raise ValueError("'<think>' occurs after '</think>'.")
    think_body = assistant[start:end]
    # literal "<think>" must NOT occur inside block
    if "<think>" in think_body:
        raise ValueError("Literal '<think>' found inside the <think>...</think> block content.")
    return think_body, start, end

def take_first_json_array_prefix(txt: str):
    """Given the think_body, extract the leading JSON array using bracket depth."""
    s = txt.lstrip()
    if not s.startswith("["):
        raise ValueError("Think(A) must start with a JSON array '[' right after <think> (allowing whitespace).")
    depth = 0
    for i, ch in enumerate(s):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                arr_text = s[:i+1]
                rest = s[i+1:]
                # Validate JSON
                try:
                    arr = json.loads(arr_text)
                except Exception as e:
                    raise ValueError(f"Think(A) array is not valid JSON: {e}")
                return arr, arr_text, rest
        elif ch in "{":
            depth += 1
        elif ch in "}":
            depth -= 1
    raise ValueError("Could not find a balanced JSON array in think(A).")

def check_label_line(rest_after_array: str):
    """
    From the remainder of the think body, find the label line (C) with EM DASH.
    Return (ok, conflict_type, conflict_reason, consumed_upto_index).
    """
    lines = rest_after_array.strip("\n").splitlines()
    if not lines:
        raise ValueError("Think(B/C/D) content missing after the JSON array.")
    # Find first line containing EM DASH exactly
    label_idx = None
    for idx, line in enumerate(lines):
        if " — " in line:
            label_idx = idx
            break
    if label_idx is None:
        raise ValueError("Did not find label line with EM DASH ' — '.")
    label_line = lines[label_idx].strip()
    # Parse label
    parts = label_line.split(" — ", 1)
    if len(parts) != 2:
        raise ValueError("Malformed label line around EM DASH.")
    conflict_type, reason = parts[0].strip(), parts[1].strip()
    if conflict_type not in CONFLICT_TYPES:
        raise ValueError(f"Conflict type not recognized on label line: {conflict_type!r}")
    if word_count(reason) > 50:
        raise ValueError(f"Label line reason exceeds 50 words (got {word_count(reason)}).")
    # Return content indices to allow checking (B) and (D)
    before_label = "\n".join(lines[:label_idx])
    after_label = "\n".join(lines[label_idx+1:])
    return conflict_type, reason, before_label, after_label

def has_named_mechanism(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in MECHANISMS)

def extract_final_answer(assistant: str, end_idx: int):
    # after </think> there must be exactly ONE blank line, then the answer (up to sentinel)
    post = assistant[end_idx + len("</think>"):]
    post = re.sub(r"\r\n?", "\n", post)
    if not post.startswith("\n\n"):
        raise ValueError("There must be exactly one blank line after </think> before the final answer.")
    body = post[2:]
    if "[[END-OF-ANSWER]]" not in body:
        raise ValueError("Assistant answer is missing the sentinel '[[END-OF-ANSWER]]'.")
    answer = body.rsplit("[[END-OF-ANSWER]]", 1)[0].rstrip()
    return answer

def check_citations(answer: str, n_docs: int):
    # Split into sentences (approx)
    sents = [s for s in re.split(r'(?<=[\.!?])\s+', answer.strip()) if s]
    if not sents:
        raise ValueError("Final answer seems empty.")
    abstain = (answer.strip() == "CANNOT ANSWER, INSUFFICIENT EVIDENCE")
    if abstain:
        return
    if not (2 <= len(sents) <= 4):
        raise ValueError(f"Final answer should have 2–4 sentences (found {len(sents)}).")
    pat = re.compile(r"\[d(\d+)\]")
    with_cite = 0
    cited_ids = set()
    for s in sents:
        ids = [int(m) for m in pat.findall(s)]
        if ids:
            with_cite += 1
            cited_ids.update(ids)
    if with_cite / len(sents) < 0.8:
        raise ValueError(f"≥80% of sentences must include [dK] citations; got {with_cite}/{len(sents)}.")
    oob = [k for k in cited_ids if k < 1 or k > n_docs]
    if oob:
        raise ValueError(f"Found out-of-bounds citations [d{k}] not in 1..{n_docs}: {sorted(oob)}.")

def validate_file(path: str, system_tpl: str, user_tpl: str, user_suffix: str):
    errors = []
    counts = 0
    sys_eq = 0
    user_suffix_eq = 0

    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            counts += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                errors.append((ln, f"JSON parse error: {e}"))
                continue

            # Basic structure
            if "messages" not in obj or not isinstance(obj["messages"], list) or len(obj["messages"]) != 3:
                errors.append((ln, "Object must have messages array of length 3 (system,user,assistant)."))
                continue
            m_sys, m_user, m_asst = obj["messages"]

            if m_sys.get("role") != "system":
                errors.append((ln, "First message role must be 'system'."))
                continue
            if m_user.get("role") != "user":
                errors.append((ln, "Second message role must be 'user'."))
                continue
            if m_asst.get("role") != "assistant":
                errors.append((ln, "Third message role must be 'assistant'."))
                continue

            sys_content = norm(m_sys.get("content",""))
            user_content = m_user.get("content","")
            asst_content = m_asst.get("content","")

            # System must match exactly
            if sys_content != norm(system_tpl):
                errors.append((ln, "System prompt content does not match provided v5 system template."))
            else:
                sys_eq += 1

            # User anchors & suffix check
            try:
                qstr, retrieved_txt, notes_txt, user_task_suffix = split_user_sections(user_content)
            except Exception as e:
                errors.append((ln, f"User message anchor/structure error: {e}"))
                continue

            if norm(user_task_suffix) != norm(user_suffix):
                errors.append((ln, "User message 'Task:' block does not match v5 user template suffix exactly."))
            else:
                user_suffix_eq += 1

            # Parse retrieved_docs & per_doc_notes JSON
            try:
                retrieved = parse_json_block(retrieved_txt, "retrieved_docs")
                notes = parse_json_block(notes_txt, "per_doc_notes")
            except Exception as e:
                errors.append((ln, str(e)))
                continue
            if not isinstance(retrieved, list):
                errors.append((ln, "retrieved_docs must be a JSON list."))
                continue
            n_docs = len(retrieved)

            # Assistant: think block
            try:
                think_body, tstart, tend = find_think_block(asst_content)
            except Exception as e:
                errors.append((ln, f"Think tag error: {e}"))
                continue

            # one blank line between </think> and answer; & sentinel
            try:
                final_answer = extract_final_answer(asst_content, tend)
            except Exception as e:
                errors.append((ln, f"Post-think answer error: {e}"))
                continue

            # (A) first JSON array
            try:
                arr, arr_text, rest = take_first_json_array_prefix(think_body)
            except Exception as e:
                errors.append((ln, f"Think(A) error: {e}"))
                continue

            if not isinstance(arr, list) or len(arr) != n_docs:
                errors.append((ln, f"Think(A) array length {len(arr)} must equal number of retrieved docs {n_docs}."))
                continue

            # Validate doc order and fields
            for idx, item in enumerate(arr, start=1):
                if not isinstance(item, dict):
                    errors.append((ln, f"Think(A) item #{idx} is not an object."))
                    break
                did = item.get("doc_id")
                if did != f"d{idx}":
                    errors.append((ln, f"Think(A) doc_id at position {idx} must be 'd{idx}' (got {did!r})."))
                    break
                verdict = item.get("verdict")
                vreason = item.get("verdict_reason","")
                kf = item.get("key_fact","")
                qual = item.get("source_quality")
                if verdict not in ALLOWED_VERDICTS:
                    errors.append((ln, f"Think(A) verdict must be one of {sorted(ALLOWED_VERDICTS)} (got {verdict!r})."))
                    break
                if qual not in ALLOWED_QUALITY:
                    errors.append((ln, f"Think(A) source_quality must be one of {sorted(ALLOWED_QUALITY)} (got {qual!r})."))
                    break
                if verdict == "irrelevant" and kf != "":
                    errors.append((ln, "Think(A) if verdict=='irrelevant' then key_fact must be empty string."))
                    break
                if verdict != "irrelevant" and word_count(kf) > 80:
                    errors.append((ln, f"Think(A) key_fact exceeds 80 words (got {word_count(kf)})."))
                    break
                if word_count(vreason) > 80:
                    errors.append((ln, f"Think(A) verdict_reason exceeds 80 words (got {word_count(vreason)})."))
                    break

            # (B/C/D)
            try:
                conflict_type, reason, before_label, after_label = check_label_line(rest)
            except Exception as e:
                errors.append((ln, f"Think(C) label error: {e}"))
                continue

            # (B) must reference doc ids and name mechanism
            if not re.search(r"\bd\d+\b", before_label):
                errors.append((ln, "Think(B) should reference specific doc IDs (e.g., d1, d2)."))
            if not has_named_mechanism(before_label):
                errors.append((ln, "Think(B) must NAME one mechanism: "
                                   + ", ".join(sorted(MECHANISMS)) + "."))

            # Final answer checks
            try:
                check_citations(final_answer, n_docs)
            except Exception as e:
                errors.append((ln, f"Final answer citation/length error: {e}"))

    return counts, errors, sys_eq, user_suffix_eq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", required=True, help="Directory containing *_final.jsonl")
    ap.add_argument("--system_file", required=True, help="system_text_mode_sft.v5.txt")
    ap.add_argument("--user_file", required=True, help="user_text_mode_sft.v5.txt")
    args = ap.parse_args()

    system_tpl = load_template(args.system_file)
    user_tpl = load_template(args.user_file)
    user_suffix = extract_user_template_suffix(user_tpl)

    files = [
        os.path.join(args.splits_dir, "train.text.messages_v5_final.jsonl"),
        os.path.join(args.splits_dir, "val.text.messages_v5_final.jsonl"),
        os.path.join(args.splits_dir, "test.text.messages_v5_final.jsonl"),
    ]

    grand_total = 0
    grand_errors = 0
    for fp in files:
        if not os.path.isfile(fp):
            print(f"[WARN] Missing file: {fp}")
            continue
        total, errors, sys_ok, user_ok = validate_file(fp, system_tpl, user_tpl, user_suffix)
        grand_total += total
        grand_errors += len(errors)
        print(f"\n=== Validation: {fp} ===")
        print(f"Rows: {total}")
        print(f"System-prompt exact matches: {sys_ok}/{total}")
        print(f"User 'Task:' suffix exact matches: {user_ok}/{total}")
        print(f"Rows with any errors: {len(errors)}")
        kind_counter = Counter([e[1] for e in errors])
        if kind_counter:
            print("Distinct error kinds:")
            for k, v in kind_counter.most_common():
                print(f"  {v:>4} × {k}")
        for ln, msg in errors[:10]:
            print(f"  [line {ln}] {msg}")

    print("\n=== Overall Summary ===")
    print(f"Total rows checked: {grand_total}")
    print(f"Total rows with errors: {grand_errors}")

if __name__ == "__main__":
    main()