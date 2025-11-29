# scripts/validate_normalized.py
import json, sys, re

def ok_ts(ts: str) -> bool:
    if ts == "": return True
    if re.fullmatch(r"\d{4}", ts): return True
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", ts): return True
    return False

def main(path: str):
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            obj = json.loads(line)
            if not obj.get("id", "").startswith("#"):
                print(f"[{i}] bad id: {obj.get('id')}")
                bad += 1
            if not obj.get("query"):
                print(f"[{i}] missing query")
                bad += 1
            rds = obj.get("retrieved_docs", [])
            if not isinstance(rds, list) or not rds:
                print(f"[{i}] empty retrieved_docs")
                bad += 1
            else:
                for d in rds:
                    if not d.get("doc_id", "").startswith("d"):
                        print(f"[{i}] bad doc_id: {d.get('doc_id')}")
                        bad += 1
                    if "source_url" not in d:
                        print(f"[{i}] missing source_url")
                        bad += 1
                    if not isinstance(d.get("snippet", ""), str) or d["snippet"] == "":
                        print(f"[{i}] empty snippet")
                        bad += 1
                    if not ok_ts(d.get("timestamp", "")):
                        print(f"[{i}] bad timestamp: {d.get('timestamp')}")
                        bad += 1
            if "conflict_type" not in obj:
                print(f"[{i}] missing conflict_type")
                bad += 1
            if "gold_answer" not in obj:
                print(f"[{i}] missing gold_answer")
                bad += 1
    print("bad =", bad)
    return bad

if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))