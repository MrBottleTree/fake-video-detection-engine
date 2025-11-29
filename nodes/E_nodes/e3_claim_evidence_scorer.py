import logging
from typing import List, Dict, Any
from nodes import dump_node_debug

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_verdict(score: float) -> str:
    """Determines the verdict based on the aggregated score."""
    if score > 0.7:
        return "Highly Likely"
    elif score > 0.4:
        return "Likely"
    elif score > 0.2:
        return "Possible"
    else:
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

    # Group evidence by claim_id and claim_text
    evidence_map_by_id = {}
    evidence_map_by_text = {}
    for ev in evidence_list:
        cid = ev.get("claim_id")
        ctext = ev.get("claim_text")
        if cid:
            evidence_map_by_id.setdefault(cid, []).append(ev)
        if ctext:
            evidence_map_by_text.setdefault(ctext, []).append(ev)
    evidence_text_keys = list(evidence_map_by_text.keys())
    global_evidence_avg = 0.0
    if evidence_list:
        global_evidence_avg = sum(ev.get("reliability_score", 0.0) for ev in evidence_list) / max(len(evidence_list), 1)
            
    scored_claims = []
    for claim_obj in claims_data:
        # Handle both string and dict claims in the loop (though we normalized above)
        claim_text = None
        if isinstance(claim_obj, dict):
            claim_text = claim_obj.get("claim_text") or claim_obj.get("text")
            claim_id = claim_obj.get("id")
        else:
            claim_text = str(claim_obj)
            claim_id = None
        if not claim_text:
            continue
        
        related_evidence = []
        if claim_id and claim_id in evidence_map_by_id:
            related_evidence = evidence_map_by_id[claim_id]
        elif claim_text in evidence_map_by_text:
            related_evidence = evidence_map_by_text[claim_text]
        else:
            # Loose match: substring/contains to salvage evidence alignment
            lowered = claim_text.lower()
            for key in evidence_text_keys:
                lk = key.lower()
                if lowered in lk or lk in lowered:
                    related_evidence.extend(evidence_map_by_text.get(key, []))
        evidence_count = len(related_evidence)

        final_score = 0.0
        verdict = "Unverified"
        
        if evidence_count > 0:
            # Calculate average reliability score
            # We assume all evidence supports the claim for now
            total_reliability = sum(e.get("reliability_score", 0.0) for e in related_evidence)
            avg_reliability = total_reliability / evidence_count
            
            # Optional: Small boost for quantity (logarithmic-ish)
            # If we have many sources, confidence should go up, even if they are medium reliability.
            # But E2 already boosted for >2 sources.
            # Let's just use the average for now to be safe and not over-inflate.
            
            final_score = avg_reliability
            verdict = get_verdict(final_score)
        elif global_evidence_avg > 0:
            # Fallback: use global average reliability when no exact/loose match was found
            final_score = max(final_score, global_evidence_avg * 0.5)
            verdict = get_verdict(final_score)
            evidence_count = len(evidence_list)
        
        # Create new claim object with results
        new_claim = claim_obj.copy() if isinstance(claim_obj, dict) else {"text": claim_text}
        new_claim.setdefault("text", claim_text)
        new_claim.setdefault("claim_text", claim_text)
        new_claim.update({
            "evidence_score": round(final_score, 2),
            "verdict": verdict,
            "evidence_count": evidence_count
        })
        
        scored_claims.append(new_claim)
        safe_claim_text = claim_text if claim_text else ""
        print(f"Claim: '{safe_claim_text}...' -> Verdict: {verdict} (Score: {final_score:.2f}, Sources: {evidence_count})")

    state["claims"] = scored_claims
    dump_node_debug(
        state,
        "E3",
        {
            "claims_scored": len(scored_claims),
            "avg_score": sum(c.get("evidence_score", 0) for c in scored_claims) / max(len(scored_claims), 1),
        },
    )
    return state
