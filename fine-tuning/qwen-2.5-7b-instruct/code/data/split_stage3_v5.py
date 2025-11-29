#!/usr/bin/env python3
import json, argparse, random
from collections import defaultdict, Counter
from pathlib import Path

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def largest_remainder_counts(n, ratios):
    # ratios is a tuple/list (r_train, r_val, r_test), sum ~ 1.0
    raw = [n * r for r in ratios]
    base = [int(x) for x in raw]
    rem = n - sum(base)
    # assign remaining by largest fractional parts
    fracs = [(raw[i] - base[i], i) for i in range(3)]
    fracs.sort(reverse=True)
    for k in range(rem):
        base[fracs[k][1]] += 1
    return tuple(base)  # (train, val, test)

def stratified_split(rows, label_key, ratios, seed=42):
    rnd = random.Random(seed)
    buckets = defaultdict(list)
    for r in rows:
        ct = r.get(label_key, "")
        ct = ct if isinstance(ct, str) else ""
        buckets[ct.strip()].append(r)

    train, val, test = [], [], []
    for ct_label, items in buckets.items():
        rnd.shuffle(items)
        n = len(items)
        n_tr, n_va, n_te = largest_remainder_counts(n, ratios)
        train.extend(items[:n_tr])
        val.extend(items[n_tr:n_tr+n_va])
        test.extend(items[n_tr+n_va:])
    # Final shuffle to avoid label-wise blocks (stable, reproducible)
    rnd.shuffle(train); rnd.shuffle(val); rnd.shuffle(test)
    return train, val, test

def summarize(split_name, rows):
    ct = Counter((r.get("conflict_type") or "").strip() for r in rows)
    total = len(rows)
    parts = [f"{split_name}: {total}"]
    for k,v in ct.most_common():
        if not k: continue
        pct = 100.0 * v / total if total else 0.0
        parts.append(f"  - {k}: {v} ({pct:.1f}%)")
    return "\n".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", default="data/raw/stage3_final.jsonl")
    ap.add_argument("--out_dir", default="data/splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                    help="train val test ratios (sum≈1.0)")
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    out_dir = Path(args.out_dir)

    all_rows = list(read_jsonl(in_path))

    # Keep everything except rows with empty conflict_type (exactly your requirement)
    kept = []
    skipped_empty_ct = 0
    for r in all_rows:
        ct = r.get("conflict_type")
        if isinstance(ct, str) and ct.strip() != "":
            kept.append(r)
        else:
            skipped_empty_ct += 1

    train, val, test = stratified_split(kept, "conflict_type", args.ratios, seed=args.seed)

    write_jsonl(out_dir / "train_v5.jsonl", train)
    write_jsonl(out_dir / "val_v5.jsonl",   val)
    write_jsonl(out_dir / "test_v5.jsonl",  test)

    print(f"[Split v5] file={in_path}")
    print(f"  total={len(all_rows)}  kept={len(kept)}  skipped_empty_conflict_type={skipped_empty_ct}")
    print(f"  ratios={tuple(args.ratios)}  seed={args.seed}")
    print()
    print(summarize("train_v5", train))
    print(summarize("val_v5",   val))
    print(summarize("test_v5",  test))
    print(f"\n→ wrote to {out_dir}")

if __name__ == "__main__":
    main()