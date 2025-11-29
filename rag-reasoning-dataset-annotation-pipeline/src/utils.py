# src/utils.py
from __future__ import annotations

import json
import re
from typing import Iterable, List, Tuple, Dict, Any, Optional
from urllib.parse import urlparse
from pathlib import Path

# -------------------------------------------------------------------
# Text helpers
# -------------------------------------------------------------------

_WS = re.compile(r"\s+")
_SENT_SPLIT = re.compile(r"([.!?]+)")
_JSON_FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)

def make_record_id(idx: int) -> str:
    """#0001, #0002, ... (1-based index)."""
    return f"#{idx:04d}"

def norm_ws(s: str) -> str:
    return _WS.sub(" ", (s or "").strip())

def word_count(s: str) -> int:
    return 0 if not s else len(re.findall(r"\b\w+\b", s))

# -------------------------------------------------------------------
# Source quality classification (only "high" or "low")
# -------------------------------------------------------------------

_HIGH_TLDS = (".gov", ".edu")
_HIGH_HOSTS = {
    # International orgs / public health
    "who.int", "un.org", "cdc.gov", "nih.gov", "europa.eu", "ec.europa.eu",
    # Peer-reviewed journals / scholarly publishers
    "nature.com", "science.org", "thelancet.com", "nejm.org", "jamanetwork.com",
    # Reference sites
    "britannica.com",
    # Major newswires / newspapers
    "reuters.com", "bbc.com", "apnews.com", "nytimes.com", "wsj.com",
    "theguardian.com", "ft.com", "washingtonpost.com",
    # Official org sites (examples)
    "mayoclinic.org"
}

def _host(domain: str) -> str:
    domain = (domain or "").lower()
    parts = domain.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return domain

def source_quality_from_url(url: Optional[str]) -> str:
    """Return 'high' or 'low' based on URL only (no 'medium')."""
    if not url:
        return "low"
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return "low"
    if any(host.endswith(tld) for tld in _HIGH_TLDS):
        return "high"
    base = _host(host)
    if host in _HIGH_HOSTS or base in _HIGH_HOSTS:
        return "high"
    for h in _HIGH_HOSTS:
        if host.endswith(h):
            return "high"
    return "low"

# -------------------------------------------------------------------
# JSON extraction (robust against code fences and trailing commas)
# -------------------------------------------------------------------

def extract_first_json(text: str) -> Tuple[Optional[Dict[str,Any]], str]:
    if text is None:
        return None, ""
    cleaned = _JSON_FENCE.sub("", text).strip()
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not m:
        return None, cleaned
    frag = m.group(0)
    try:
        return json.loads(frag), frag
    except Exception:
        frag2 = re.sub(r",\s*}", "}", frag)
        frag2 = re.sub(r",\s*]", "]", frag2)
        try:
            return json.loads(frag2), frag2
        except Exception:
            return None, frag

# -------------------------------------------------------------------
# NEW: Robust JSON extractor for Stage-2 async parsing
# -------------------------------------------------------------------

def extract_json(text: str) -> Tuple[Optional[Dict[str,Any]], str]:
    """
    More tolerant JSON extractor used in Stage-2 async script.
    - Strips code fences and markdown.
    - Finds the first {...} block if wrapped in prose.
    - Returns (parsed_dict, raw_fragment).
    - Returns (None, '') if nothing valid is found.
    """
    if not text:
        return None, ""

    cleaned = _JSON_FENCE.sub("", text or "").strip()

    # Try direct parse first
    try:
        parsed = json.loads(cleaned)
        return parsed, cleaned
    except Exception:
        pass

    # Fallback: find the first {...} JSON block
    m = re.search(r"\{[\s\S]*?\}", cleaned)
    if not m:
        return None, cleaned
    frag = m.group(0)

    # Try parsing that fragment
    try:
        return json.loads(frag), frag
    except Exception:
        # Attempt to fix trailing commas
        frag2 = re.sub(r",\s*([\]}])", r"\1", frag)
        try:
            return json.loads(frag2), frag2
        except Exception:
            return None, frag

# -------------------------------------------------------------------
# Stage-1 validation (<= 40 words for quote & verdict_reason)
# -------------------------------------------------------------------

STAGE1_ALLOWED_VERDICTS = {"supports", "partially supports", "irrelevant"}
STAGE1_ALLOWED_QUALITY = {"high", "low"}
STAGE1_QUOTE_MAX = 50
STAGE1_REASON_MAX = 50

def validate_stage1_record(rec: Dict[str,Any]) -> List[str]:
    errs: List[str] = []
    verdict = rec.get("verdict")
    if verdict not in STAGE1_ALLOWED_VERDICTS:
        errs.append(f"invalid verdict: {verdict}")
    if verdict == "irrelevant":
        if rec.get("key_fact") or rec.get("quote"):
            errs.append("irrelevant record must have empty key_fact and quote")
    else:
        if not rec.get("key_fact"):
            errs.append("missing key_fact")
        q = rec.get("quote","")
        if not q:
            errs.append("missing quote")
        elif word_count(q) > STAGE1_QUOTE_MAX:
            errs.append(f"quote exceeds {STAGE1_QUOTE_MAX} words")
        r = rec.get("verdict_reason","")
        if word_count(r) > STAGE1_REASON_MAX:
            errs.append(f"verdict_reason exceeds {STAGE1_REASON_MAX} words")
        qual = rec.get("source_quality")
        if qual not in STAGE1_ALLOWED_QUALITY:
            errs.append(f"invalid source_quality: {qual}")
    return errs

# -------------------------------------------------------------------
# Stage-2 validation (<= 50 words conflict_reason)
# -------------------------------------------------------------------

STAGE2_REASON_MAX = 50

def validate_stage2_record(rec: Dict[str,Any]) -> List[str]:
    errs: List[str] = []
    reason = rec.get("conflict_reason","")
    if word_count(reason) > STAGE2_REASON_MAX:
        errs.append(f"conflict_reason exceeds {STAGE2_REASON_MAX} words")
    if not isinstance(rec.get("answerable_under_evidence"), bool):
        errs.append("answerable_under_evidence must be boolean")
    return errs

# -------------------------------------------------------------------
# Stage-3 helpers
# -------------------------------------------------------------------

ABSTAIN_STRING = "CANNOT ANSWER, INSUFFICIENT EVIDENCE"
_CIT_RE = re.compile(r"$begin:math:display$(d\\d+)$end:math:display$", re.IGNORECASE)

def ensure_single_think_block(s: str) -> bool:
    return len(re.findall(r"<think>.*?</think>", s or "", flags=re.DOTALL)) == 1

def parse_citations(answer: str) -> List[str]:
    return [m.group(1).lower() for m in _CIT_RE.finditer(answer or "")]

def sentence_chunks(text: str) -> List[str]:
    if not text:
        return []
    parts = _SENT_SPLIT.split(text.strip())
    out: List[str] = []
    cur = ""
    for tok in parts:
        cur += tok
        if _SENT_SPLIT.match(tok):
            if cur.strip():
                out.append(cur.strip())
            cur = ""
    if cur.strip():
        out.append(cur.strip())
    return out

def citation_coverage(answer: str) -> float:
    sents = sentence_chunks(answer)
    if not sents:
        return 0.0
    cited = [s for s in sents if _CIT_RE.search(s)]
    return len(cited) / len(sents)

def quality_map(per_doc_notes: Iterable[Dict[str,Any]]) -> Dict[str,str]:
    return {str(d.get("doc_id","")).lower(): (d.get("source_quality") or "low") for d in per_doc_notes}

def order_evidence_high_first(cited_ids: List[str], per_doc_notes: Iterable[Dict[str,Any]]) -> List[str]:
    qmap = quality_map(per_doc_notes)
    def kfn(did: str) -> Tuple[int,int]:
        return (0 if qmap.get(did,"low") == "high" else 1, cited_ids.index(did))
    known = set(qmap.keys())
    filtered = [d for d in cited_ids if d in known]
    return sorted(filtered, key=kfn)

# -------------------------------------------------------------------
# NEW: Normalized data helpers for single-query runs
# -------------------------------------------------------------------

def coerce_query_id(qid: str | int) -> str:
    """
    Accepts: 2, "2", "0002", "#2", "#0002" → returns "#0002"
    """
    if isinstance(qid, int):
        return f"#{qid:04d}"
    s = str(qid).strip()
    s = s[1:] if s.startswith("#") else s
    s = re.sub(r"^0+", "", s) or "0"
    return f"#{int(s):04d}"

def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def _read_json_or_jsonl(path: Path) -> List[Dict[str,Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if raw.startswith("["):
        return json.loads(raw)
    # JSONL / JSOLN (one JSON object per line)
    return list(_iter_jsonl(path))

def load_query_entry(normalized_path: str | Path, query_id: str | int) -> Dict[str,Any]:
    """
    Load the normalized dataset and return the entry with the matching id.
    Raises FileNotFoundError / ValueError if not found.
    """
    p = Path(normalized_path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    want = coerce_query_id(query_id)
    rows = _read_json_or_jsonl(p)
    for row in rows:
        if str(row.get("id","")).strip() == want:
            return row
    raise ValueError(f"Query id {want} not found in {p}")

def to_stage1_items_from_entry(entry: Dict[str,Any]) -> List[Dict[str,Any]]:
    """
    Convert a normalized entry to a list of Stage-1 items (one per retrieved doc).
    Each item has shape:
      { "query": <str>, "doc": {doc_id, source_url, snippet, timestamp} }
    """
    q = entry["query"]
    items: List[Dict[str,Any]] = []
    for d in entry.get("retrieved_docs", []):
        items.append({"query": q, "doc": {
            "doc_id": d["doc_id"],
            "source_url": d.get("source_url","") or "",
            "snippet": d.get("snippet","") or "",
            "timestamp": d.get("timestamp","") or "",
        }})
    return items

def output_path_stage1(base_dir: str | Path, query_id: str | int) -> Path:
    """
    Build data/stage1_outputs/stage1_q000X.jsonl (avoid '#' in file names).
    """
    pid = coerce_query_id(query_id)   # "#0002"
    digits = pid[1:]                  # "0002"
    out_dir = Path(base_dir) / "stage1_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"stage1_q{digits}.jsonl"

def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)