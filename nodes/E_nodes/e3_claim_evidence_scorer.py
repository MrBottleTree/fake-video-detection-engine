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
    """
    E3 Node: Claim Evidence Scorer
    
    Aggregates the reliability scores of evidence to determine a final
    score and verdict for each claim.
    """
    print("--- E3: Claim Evidence Scorer ---")
    
    # REQUIRED CHANGE: Use the grouped evidence structure from E1/E2
    evidence_list = state.get("evidence", [])
    
    if not evidence_list:
        print("No claims found to score.")
        return state

    scored_claims = []
    
    # REQUIRED CHANGE: Iterate directly over the grouped evidence
    for entry in evidence_list:
        # Extract claim info from the grouped entry
        claim_obj = entry.get("claim", {})
        
        # Handle case where claim might be a dict (from E1) or string
        if isinstance(claim_obj, dict):
            claim_text = claim_obj.get("claim_text", str(claim_obj))
        else:
            claim_text = str(claim_obj)

        # REQUIRED CHANGE: Get results directly from the entry
        related_evidence = entry.get("results", [])
        evidence_count = len(related_evidence)
        
        final_score = 0.0
        verdict = "Unverified"
        
        if evidence_count > 0:
            # Calculate average reliability score
            total_reliability = sum(e.get("reliability_score", 0.0) for e in related_evidence)
            final_score = total_reliability / evidence_count
            verdict = get_verdict(final_score)
        
        # Create new claim object with results, preserving original fields if possible
        new_claim = claim_obj.copy() if isinstance(claim_obj, dict) else {"text": claim_text}
        
        # Ensure 'text' key exists for downstream compatibility
        if "text" not in new_claim:
            new_claim["text"] = claim_text

        new_claim.update({
            "evidence_score": round(final_score, 2),
            "verdict": verdict,
            "evidence_count": evidence_count
        })
        
        scored_claims.append(new_claim)
        
        # Logging output similar to original
        safe_claim_text = claim_text if claim_text else ""
        print(f"Claim: '{safe_claim_text[:50]}...' -> Verdict: {verdict} (Score: {final_score:.2f}, Sources: {evidence_count})")

    state["claims"] = scored_claims