import uuid
from typing import Any, Dict, List, Optional


def normalize_claims(claims_raw: List[Any]) -> List[Dict[str, Any]]:
    """Ensure every claim is a dict with id and claim_text/text fields."""
    normalized: List[Dict[str, Any]] = []
    for c in claims_raw or []:
        claim_id = str(uuid.uuid4())
        if isinstance(c, dict):
            if "id" in c:
                claim_id = str(c["id"])
            claim_text = c.get("claim_text") or c.get("text") or str(c)
            normalized.append(
                {
                    "id": claim_id,
                    "claim_text": claim_text,
                    "text": claim_text,
                    "who": c.get("who"),
                    "what": c.get("what"),
                    "when": c.get("when"),
                    "where": c.get("where"),
                    "source": c.get("source"),
                    "confidence": c.get("confidence"),
                }
            )
        else:
            claim_text = str(c)
            normalized.append(
                {
                    "id": claim_id,
                    "claim_text": claim_text,
                    "text": claim_text,
                    "who": None,
                    "what": None,
                    "when": None,
                    "where": None,
                    "source": None,
                    "confidence": None,
                }
            )
    return normalized


def flatten_evidence(evidence_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize evidence into a flat list.
    Supports both legacy structure (per-claim objects with nested 'results')
    and the flattened structure produced by the updated E1.
    """
    flat: List[Dict[str, Any]] = []
    for item in evidence_list or []:
        if isinstance(item, dict) and "results" in item:
            claim_meta = item.get("claim", {})
            claim_text = claim_meta.get("claim_text") or claim_meta.get("text")
            claim_id = claim_meta.get("id")
            for idx, res in enumerate(item.get("results") or [], start=1):
                entry = res.copy()
                entry.setdefault("rank", idx)
                if claim_text:
                    entry.setdefault("claim_text", claim_text)
                if claim_id:
                    entry.setdefault("claim_id", claim_id)
                flat.append(entry)
        else:
            flat.append(item)
    return flat
