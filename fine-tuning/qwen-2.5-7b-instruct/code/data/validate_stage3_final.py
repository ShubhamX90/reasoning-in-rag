#!/usr/bin/env python3
import json, re, argparse
from collections import Counter, defaultdict
from pathlib import Path

DOC_ID_RE = re.compile(r"^d[1-9]\d*$")

CONTRACT_TYPES = {
    "No conflict",
    "Complementary information",
    "Conflicting opinions or research outcomes",
    "Conflict due to outdated information",
    "Conflict due to misinformation",
}

def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for ln, s in enumerate(f, 1):
            s = s.strip()
            if not s:
                continue
            try:
                yield ln, json.loads(s)
            except Exception as e:
                yield ln, {"__parse_error__": str(e)}

def check_doc_id_sequence(doc_ids):
    # must be d1..dN contiguous and ordered
    if not doc_ids:
        return False, "no retrieved_docs"
    expected = [f"d{i+1}" for i in range(len(doc_ids))]
    return (doc_ids == expected, f"doc_id sequence {doc_ids} != {expected}")

def validate_one(obj):
    errs = []

    # JSON parse guard
    if "__parse_error__" in obj:
        errs.append(f"bad_json: {obj['__parse_error__']}")
        return errs

    # core fields
    if not isinstance(obj.get("id"), str) or not obj.get("id"):
        errs.append("field:id missing_or_empty")
    if not isinstance(obj.get("query"), str) or not obj.get("query"):
        errs.append("field:query missing_or_empty")

    ct = obj.get("conflict_type")
    if ct is None:
        errs.append("field:conflict_type missing")
    elif not isinstance(ct, str):
        errs.append("field:conflict_type not_string")
    elif ct.strip() == "":
        errs.append("conflict_type empty")

    # retrieved_docs
    rd = obj.get("retrieved_docs")
    if not isinstance(rd, list) or len(rd) == 0:
        errs.append("retrieved_docs missing_or_empty")
        rd = []

    doc_ids = []
    for i, d in enumerate(rd):
        if not isinstance(d, dict):
            errs.append(f"retrieved_docs[{i}] not_object")
            continue
        did = d.get("doc_id")
        if not isinstance(did, str) or not DOC_ID_RE.match(did):
            errs.append(f"retrieved_docs[{i}].doc_id invalid")
        else:
            doc_ids.append(did)
        if "snippet" not in d or not isinstance(d.get("snippet"), str):
            errs.append(f"retrieved_docs[{i}].snippet missing_or_not_string")

    # d1..dN sequence
    ok, msg = check_doc_id_sequence(doc_ids)
    if not ok:
        errs.append(f"retrieved_docs.doc_id_sequence bad: {msg}")

    # per_doc_notes
    notes = obj.get("per_doc_notes")
    if not isinstance(notes, list):
        errs.append("per_doc_notes missing_or_not_list")
        notes = []
    note_map = {n.get("doc_id"): n for n in notes if isinstance(n, dict)}
    for did in doc_ids:
        if did not in note_map:
            errs.append(f"per_doc_notes missing_for:{did}")

    for i, n in enumerate(notes):
        if not isinstance(n, dict):
            errs.append(f"per_doc_notes[{i}] not_object")
            continue
        did = n.get("doc_id")
        verdict = n.get("verdict")
        if verdict not in {"supports", "partially supports", "irrelevant"}:
            errs.append(f"per_doc_notes[{did or i}].verdict invalid:{verdict}")
        kf = n.get("key_fact", "")
        if verdict == "irrelevant" and kf not in ("", None):
            errs.append(f"per_doc_notes[{did}].key_fact must_be_empty_when_irrelevant")
        if verdict != "irrelevant" and (not isinstance(kf, str) or not kf.strip()):
            errs.append(f"per_doc_notes[{did}].key_fact missing_for_supporting_verdict")
        vr = n.get("verdict_reason", "")
        if not isinstance(vr, str) or not vr.strip():
            errs.append(f"per_doc_notes[{did}].verdict_reason missing")
        sq = (n.get("source_quality") or "").lower()
        if sq not in {"high", "low"}:
            errs.append(f"per_doc_notes[{did}].source_quality invalid:{sq}")
        # NOTE: no length checks for quote/verdict_reason anymore
        # q = n.get("quote", "")  # allowed any length/type if present

    # conflict_reason (no length checks anymore)
    # just allow missing or any length string
    cr = obj.get("conflict_reason", None)
    if cr is not None and not isinstance(cr, str):
        errs.append("conflict_reason not_string")

    # expected_response
    er = obj.get("expected_response", {})
    if not isinstance(er, dict):
        errs.append("expected_response missing_or_not_object")
        er = {}
    abstain = er.get("abstain")
    if not isinstance(abstain, bool):
        errs.append("expected_response.abstain missing_or_not_bool")
    ev = er.get("evidence", [])
    if ev is not None and not isinstance(ev, list):
        errs.append("expected_response.evidence not_list")
        ev = []
    if isinstance(ev, list):
        for eid in ev:
            if eid not in doc_ids:
                errs.append(f"expected_response.evidence unknown_id:{eid}")

    # answerable_under_evidence
    aue = obj.get("answerable_under_evidence")
    if aue is None or not isinstance(aue, bool):
        errs.append("answerable_under_evidence missing_or_not_bool")

    return errs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", default="data/raw/stage3_final.jsonl")
    ap.add_argument("--max_examples", type=int, default=15, help="max error examples to print per error type")
    args = ap.parse_args()

    totals = 0
    empty_ct = 0
    ct_counter = Counter()
    error_buckets = defaultdict(list)

    for ln, obj in read_jsonl(args.in_jsonl):
        totals += 1
        if "__parse_error__" in obj:
            error_buckets["bad_json"].append((ln, None))
            continue

        ct = obj.get("conflict_type")
        if isinstance(ct, str):
            if ct.strip() == "":
                empty_ct += 1
            else:
                ct_counter[ct.strip()] += 1

        errs = validate_one(obj)
        for e in errs:
            error_buckets[e].append((ln, obj.get("id")))

    print("=== Stage-3 JSONL Validation Summary ===")
    print(f"File: {args.in_jsonl}")
    print(f"Total rows: {totals}")
    print(f"Non-empty conflict_type rows: {sum(ct_counter.values())}")
    print(f"Empty conflict_type rows: {empty_ct}")
    print("\nConflict type distribution (non-empty):")
    for k, v in ct_counter.most_common():
        print(f"  {k}: {v}")

    total_err_rows = len(set(ln for errs in error_buckets.values() for (ln, _) in errs))
    print(f"\nRows with any validation errors: {total_err_rows}")
    print(f"Distinct error kinds: {len(error_buckets)}")

    for kind, items in sorted(error_buckets.items(), key=lambda x: (-len(x[1]), x[0])):
        print(f"\n[Error] {kind}  count={len(items)}")
        for ln, _id in items[:args.max_examples]:
            print(f"  line={ln} id={_id}")

if __name__ == "__main__":
    main()