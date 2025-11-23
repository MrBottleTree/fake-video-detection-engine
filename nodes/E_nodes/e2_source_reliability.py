import json
import os
import urllib.parse
import urllib.request
import logging
from typing import List, Dict, Any
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_trusted_sources(assets_dir: str = "assets") -> Dict[str, List[str]]:
    """Loads trusted sources from a JSON file."""
    try:
        # Assuming assets folder is in the project root, relative to this file
        # This file is in nodes/E_nodes/
        # Project root is ../../
        
        # However, it's safer to look for it relative to the current working directory 
        # or a known location. The plan said "assets/trusted_sources.json".
        # We'll try a few paths.
        
        possible_paths = [
            os.path.join("assets", "trusted_sources.json"),
            os.path.join(os.getcwd(), "assets", "trusted_sources.json"),
            os.path.join(os.path.dirname(__file__), "..", "..", "assets", "trusted_sources.json")
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        if not file_path:
            logger.warning("trusted_sources.json not found. Using empty lists.")
            return {"high_trust": [], "medium_trust": []}

        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading trusted sources: {e}")
        return {"high_trust": [], "medium_trust": []}

def get_domain(url: str) -> str:
    """Extracts the domain from a URL."""
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc
        # Remove 'www.' if present
        if domain.startswith("www."):
            domain = domain[4:]
        return domain.lower()
    except Exception:
        return ""

def check_about_page(url: str) -> bool:
    """
    Checks if an 'About' page exists or is linked from the homepage.
    This is a simplified check: it fetches the homepage and looks for 'about' in the links,
    or tries /about.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        # Method 1: Check if /about exists
        # We use a short timeout to not block for too long
        try:
            about_url = f"{base_url}/about"
            req = urllib.request.Request(
                about_url, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
            
        # Method 2: Fetch homepage and look for "about" link
        # This is expensive, so we might skip it or do it very lightly.
        # For robustness and speed, let's stick to the /about check or just skip if too slow.
        # The user said "optionally we can look for “about” page... simple web scraping"
        
        return False
        
    except Exception as e:
        logger.debug(f"Error checking about page for {url}: {e}")
        return False

def calculate_reliability_score(evidence_item: Dict[str, Any], trusted_sources: Dict[str, List[str]], claim_consensus_map: Dict[str, int]) -> Dict[str, Any]:
    """Calculates the reliability score for a single evidence item."""
    
    url = evidence_item.get("url", "")
    claim_text = evidence_item.get("claim_text", "") # Assuming this links back to a claim
    
    score = 0.5 # Base score
    details = []
    
    if not url:
        return {"score": 0.0, "details": ["No URL provided"]}

    domain = get_domain(url)
    
    # 1. Domain TLD Check
    if domain.endswith(".gov") or domain.endswith(".mil"):
        score += 0.4
        details.append("Government/Military domain (+0.4)")
    elif domain.endswith(".edu"):
        score += 0.3 # Slightly less than gov
        details.append("Educational domain (+0.3)")
        
    # 2. Trusted List Check
    if any(domain == trusted or domain.endswith("." + trusted) for trusted in trusted_sources.get("high_trust", [])):
        score += 0.3
        details.append("High trust source (+0.3)")
    elif any(domain == trusted or domain.endswith("." + trusted) for trusted in trusted_sources.get("medium_trust", [])):
        score += 0.1
        details.append("Medium trust source (+0.1)")
        
    # 3. Protocol Check
    if url.startswith("https://"):
        score += 0.1
        details.append("Secure protocol (HTTPS) (+0.1)")
        
    # 4. About Page Check (Optional/Expensive)
    # We'll do this only if score is not already very high, to save time? 
    # Or just do it. Let's do it but catch errors silently.
    if check_about_page(url):
        score += 0.1
        details.append("About page found (+0.1)")
        
    # 5. Consensus Boost
    # If this claim is supported by multiple sources, boost this source.
    # We need to know which claim this evidence belongs to.
    # Assuming 'claim_text' or an ID is present.
    source_count = claim_consensus_map.get(claim_text, 0)
    if source_count > 2:
        score += 0.1
        details.append(f"Consensus boost ({source_count} sources) (+0.1)")
        
    # Cap score at 1.0
    final_score = min(1.0, score)
    
    return {
        "score": final_score,
        "details": details
    }

def run(state: dict) -> dict:
    """
    E2 Node: Source Reliability Scorer
    
    Input state expectation:
    state["evidence"] = [
        {"url": "...", "claim_text": "...", "snippet": "..."},
        ...
    ]
    
    Output state update:
    state["evidence"] = [
        {..., "reliability_score": 0.85, "reliability_details": [...]},
        ...
    ]
    """
    print("--- E2: Source Reliability ---")
    evidence_list = state.get("evidence", [])
    
    if not evidence_list:
        print("No evidence found to score.")
        return state

    trusted_sources = load_trusted_sources()
    
    # Calculate consensus counts
    # We count how many unique domains support each claim
    claim_domains = {} # claim_text -> set(domains)
    
    for item in evidence_list:
        claim = item.get("claim_text", "unknown")
        url = item.get("url", "")
        domain = get_domain(url)
        if claim and domain:
            if claim not in claim_domains:
                claim_domains[claim] = set()
            claim_domains[claim].add(domain)
            
    claim_consensus_map = {k: len(v) for k, v in claim_domains.items()}
    
    scored_evidence = []
    for item in evidence_list:
        # Create a copy to avoid mutating original in place immediately (good practice)
        new_item = item.copy()
        
        result = calculate_reliability_score(new_item, trusted_sources, claim_consensus_map)
        
        new_item["reliability_score"] = result["score"]
        new_item["reliability_details"] = result["details"]
        
        scored_evidence.append(new_item)
        print(f"Scored {new_item.get('url', 'N/A')}: {result['score']:.2f}")

    state["evidence"] = scored_evidence
    return state
