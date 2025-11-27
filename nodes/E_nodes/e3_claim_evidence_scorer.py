import logging
from typing import List, Dict, Any
from statistics import median
from collections import defaultdict

from nodes.utils.schema import flatten_evidence, normalize_claims

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _domain_from_url(url: str) -> str:
    """Extract domain without scheme/www for diversity scoring."""
    try:
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc or ""
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""

def get_verdict(score: float, contra_dominance: float = 0.0) -> str:
    """
    Determines the verdict based on the aggregated score and contradiction dominance.
    Thresholds emphasize contradiction dominance and reward stronger signals:
      contra_dominance >= 0.4 -> False
      score >= 0.82 -> True
      0.65-0.82 -> Highly Likely
      0.45-0.65 -> Likely
      0.25-0.45 -> Unlikely
      else -> Unverified
    """
    if contra_dominance >= 0.4:
        return "False"
    if score >= 0.82:
        return "True"
    if score >= 0.65:
        return "Highly Likely"
    if score >= 0.45:
        return "Likely"
    if score >= 0.25:
        return "Unlikely"
    return "Unverified"

def run(state: dict) -> dict:
    print("E3: Claim Evidence Scorer")
    
    evidence_list = state.get("evidence", [])
    claims_input = state.get("claims", [])
    
    claims_data = []
    if claims_input and isinstance(claims_input[0], str):
        claims_data = [{"text": c} for c in claims_input]
    else:
        claims_data = claims_input if claims_input else []
        
    if not claims_data:
        print("No claims found to score.")
        return state

    # Group evidence by claim id/text
    evidence_map = {}
    for ev in evidence_list:
        claim_key = ev.get("claim_id") or ev.get("claim_text")
        if claim_key:
            evidence_map.setdefault(claim_key, []).append(ev)
            
    scored_claims = []
    for claim_obj in claims_data:
        claim_text_value = claim_obj.get("claim_text") or claim_obj.get("text")
        claim_key = claim_obj.get("id") or claim_text_value
        
        related_evidence = evidence_map.get(claim_key, []) or evidence_map.get(claim_text_value, [])
        evidence_count = len(related_evidence)
        
        final_score = 0.0
        verdict = "Unverified"
        
        if evidence_count > 0:
            reliabilities = [e.get("reliability_score", 0.0) for e in related_evidence]
            support = [e for e in related_evidence if e.get("stance", "support") != "contradict"]
            contradict = [e for e in related_evidence if e.get("stance") == "contradict"]
            
            def weighted_avg(items):
                weights = [max(0.0, e.get("reliability_score", 0.0)) for e in items]
                if not weights or sum(weights) == 0:
                    return 0.0
                # emphasize higher reliability and longer snippets/titles
                length_weights = []
                for e in items:
                    base = max(0.0, e.get("reliability_score", 0.0))
                    text_len = len((e.get("title") or "") + " " + (e.get("snippet") or ""))
                    length_boost = 1 + min(text_len / 400.0, 0.5)
                    length_weights.append(base * length_boost)
                return sum(w * w for w in length_weights) / sum(length_weights)

            support_score = weighted_avg(support)
            contra_score = weighted_avg(contradict)
            # Boost strong reliable support/contradiction
            max_support_rel = max([e.get("reliability_score", 0.0) for e in support], default=0.0)
            max_contra_rel = max([e.get("reliability_score", 0.0) for e in contradict], default=0.0)
            if max_support_rel >= 0.8:
                support_score += 0.05
            if max_contra_rel >= 0.8:
                contra_score += 0.05

            net_score = max(0.0, support_score - 1.1 * contra_score)  # heavier weight on contradiction
            if not support and contradict:
                net_score = 0.0  # pure contradiction -> no support
            contra_dominance = max(0.0, (1.2 * contra_score) - support_score)
            
            domains = {_domain_from_url(e.get("url", "")) for e in related_evidence if e.get("url")}
            diversity_score = min(1.0, len(domains) / 5) if domains else 0.0
            
            median_rel = median(reliabilities)
            
            final_score = 0.8 * net_score + 0.15 * median_rel + 0.05 * diversity_score
            final_score = max(0.0, min(1.0, final_score))
            if not support and contradict:
                final_score = 0.0
            verdict = get_verdict(final_score, contra_dominance)
        
        new_claim = claim_obj.copy() if isinstance(claim_obj, dict) else {"text": claim_text_value}
        if claim_text_value:
            new_claim.setdefault("text", claim_text_value)
            new_claim.setdefault("claim_text", claim_text_value)
        new_claim.update({
            "evidence_score": round(final_score, 2),
            "verdict": verdict,
            "evidence_count": evidence_count
        })
        
        scored_claims.append(new_claim)
        safe_claim_text = claim_text_value if claim_text_value else ""
        print(f"Claim: '{safe_claim_text[:50]}...' -> Verdict: {verdict} (Score: {final_score:.2f}, Sources: {evidence_count})")

    state["claims"] = scored_claims
    return state
