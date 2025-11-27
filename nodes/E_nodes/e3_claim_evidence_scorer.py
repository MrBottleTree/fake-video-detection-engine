import logging
from typing import List, Dict, Any

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

    # Group evidence by claim text
    evidence_map = {}
    for ev in evidence_list:
        claim_text = ev.get("claim_text")
        if claim_text:
            if claim_text not in evidence_map:
                evidence_map[claim_text] = []
            evidence_map[claim_text].append(ev)
            
    scored_claims = []
    for claim_obj in claims_data:
        # Handle both string and dict claims in the loop (though we normalized above)
        claim_text = claim_obj.get("text") if isinstance(claim_obj, dict) else str(claim_obj)
        
        related_evidence = evidence_map.get(claim_text, [])
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
        
        # Create new claim object with results
        new_claim = claim_obj.copy() if isinstance(claim_obj, dict) else {"text": claim_text}
        new_claim.update({
            "evidence_score": round(final_score, 2),
            "verdict": verdict,
            "evidence_count": evidence_count
        })
        
        scored_claims.append(new_claim)
        safe_claim_text = claim_text if claim_text else ""
        print(f"Claim: '{safe_claim_text[:50]}...' -> Verdict: {verdict} (Score: {final_score:.2f}, Sources: {evidence_count})")

    state["claims"] = scored_claims
    return state
