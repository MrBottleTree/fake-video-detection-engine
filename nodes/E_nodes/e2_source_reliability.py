import json
import os
import urllib.parse
import logging
from typing import List, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_trusted_sources() -> Dict[str, List[str]]:
    """Loads trusted sources from resources/trusted_sources.json reliably."""
    try:
        # Resolve path relative to THIS file (e2_source_reliability.py)
        # Go up two levels (../../) to reach project root, then into resources
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent
        resource_path = project_root / "resources" / "trusted_sources.json"
        
        # Fallback to assets if resources doesn't exist (for backward compatibility)
        if not resource_path.exists():
            resource_path = project_root / "assets" / "trusted_sources.json"

        if not resource_path.exists():
            logger.warning(f"trusted_sources.json not found at {resource_path}")
            return {}

        with open(resource_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading trusted sources: {e}")
        return {}

def get_domain(url: str) -> str:
    """Extracts the clean domain from a URL."""
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc
        if domain.startswith("www."):
            domain = domain[4:]
        return domain.lower()
    except Exception:
        return ""

def calculate_reliability_score(evidence_item: Dict[str, Any], trusted_sources: Dict[str, List[str]]) -> Dict[str, Any]:
    """Calculates score based on Tiers and standard metrics."""
    
    url = evidence_item.get("url", "")
    score = 0.5  # Base score for unknown sources
    details = []
    
    if not url:
        return {"score": 0.0, "details": ["No URL provided"]}

    domain = get_domain(url)
    
    # --- 1. Trusted Source Tier Check ---
    # Tier 1 & 2: High Reliability (Major News/Agencies)
    if any(domain.endswith(t) for t in trusted_sources.get("tier1", [])):
        score = 1
        details.append("Tier 1 Source (High Trust)")
    elif any(domain.endswith(t) for t in trusted_sources.get("tier2", [])):
        score = 0.8
        details.append("Tier 2 Source (Trusted Media)")
    # Tier 3: Institutions (Gov/Science)
    elif any(domain.endswith(t) for t in trusted_sources.get("tier3", [])) or domain.endswith(".gov") or domain.endswith(".mil"):
        score = 0.6
        details.append("Institutional/Gov Source")
    # Tier 4: General Tech/Biz
    elif any(domain.endswith(t) for t in trusted_sources.get("tier4", [])):
        score = 0.40
        details.append("Tier 4 Source (General)")
    # Tier 5: Low Reliability / Biased
    elif any(domain.endswith(t) for t in trusted_sources.get("tier5", [])):
        score = 0.20
        details.append("Tier 5 Source (Low Reliability/Biased)")
    # Tier 6: Satire/UGC
    elif any(domain.endswith(t) for t in trusted_sources.get("tier6", [])):
        score = 0.0
        details.append("Tier 6 Source (Satire/UGC)")
    else:
        # Fallback TLD check if not in lists
        if domain.endswith(".edu"):
            score = 0.75
            details.append("Educational Domain")
    
    # --- 2. Protocol Check ---
    if url.startswith("https://"):
        # Small boost only if score is middling
        if 0.4 <= score <= 0.8:
            score += 0.05
            details.append("HTTPS Secure")

    # Cap score
    final_score = min(1.0, max(0.0, score))
    
    return {
        "score": round(final_score, 2),
        "details": details
    }

def run(state: dict) -> dict:
    """
    E2 Node: Source Reliability Scorer
    Iterates through claims -> results and scores each URL.
    """
    print("--- E2: Source Reliability ---")
    
    # E1 outputs a list of objects: [{"claim": {...}, "results": [...]}, ...]
    evidence_data = state.get("evidence", [])
    
    if not evidence_data:
        print("No evidence data found.")
        return state

    trusted_sources = load_trusted_sources()
    
    scored_data = []
    
    # Iterate through the Claims
    for entry in evidence_data:
        # Use copy to avoid mutation issues
        new_entry = entry.copy()
        raw_results = new_entry.get("results", [])
        scored_results = []
        
        # Iterate through the Search Results for this claim
        for result_item in raw_results:
            scoring = calculate_reliability_score(result_item, trusted_sources)
            
            # Inject score into the result item
            result_item["reliability_score"] = scoring["score"]
            result_item["reliability_details"] = scoring["details"]
            
            scored_results.append(result_item)
            
            if state.get("debug"):
                print(f"  > {result_item.get('url', 'No URL')[:30]}... : {scoring['score']}")
        
        new_entry["results"] = scored_results
        scored_data.append(new_entry)

    state["evidence"] = scored_data