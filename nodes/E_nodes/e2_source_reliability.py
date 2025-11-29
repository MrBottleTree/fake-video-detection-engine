import json
import os
import urllib.parse
import urllib.request
import logging
import time
from typing import List, Dict, Any, Optional
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
from nodes import dump_node_debug

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o"
MAX_PARALLEL = max(1, int(os.getenv("E2_MAX_WORKERS", "4")))

# Initialize OpenAI client if API key is available
openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("E2: OpenAI client initialized successfully")
    except Exception as e:
        logger.warning(f"E2: Failed to initialize OpenAI client: {e}")
        openai_client = None
else:
    logger.info("E2: OPENAI_API_KEY not found, will use heuristic scoring only")

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

def evaluate_source_reliability_openai(domain: str, url: str, snippet: str, claim_text: str) -> Optional[Dict[str, Any]]:
    """
    Uses OpenAI to evaluate the reliability of a source based on multiple factors.
    
    Args:
        domain: The domain of the source
        url: The full URL of the source
        snippet: A snippet of the content from the source
        claim_text: The claim being verified
        
    Returns:
        Dict with 'score' (float 0-1) and 'reason' (str), or None if evaluation fails
    """
    if not openai_client:
        return None
        
    try:
        prompt = f"""You are an expert fact-checker evaluating source reliability. Analyze this source and provide a reliability score.

SOURCE DETAILS:
- Domain: {domain}
- URL: {url}
- Claim being verified: {claim_text}
- Content snippet: {snippet[:500]}

EVALUATION CRITERIA:
1. Domain authority and reputation (e.g., .gov, .edu, known news outlets)
2. Content quality and factual accuracy indicators
3. Presence of citations, references, or evidence
4. Objectivity vs bias indicators
5. Professionalism and credibility markers

Provide a reliability score from 0.0 (completely unreliable) to 1.0 (highly reliable).

Common examples:
- Government sites (.gov, .mil): 0.9-1.0
- Academic institutions (.edu): 0.8-0.95
- Major news outlets (Reuters, AP, BBC): 0.75-0.9
- Wikipedia: 0.7-0.8
- Personal blogs with good citations: 0.5-0.7
- Clickbait or sensationalist sites: 0.2-0.4
- Known misinformation sources: 0.0-0.2

Return ONLY a JSON object with this exact structure:
{{"score": 0.85, "reason": "Brief explanation of the score"}}"""

        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            timeout=20.0,
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        
        # Validate the response
        if "score" not in result or not isinstance(result["score"], (int, float)):
            logger.warning(f"E2: Invalid OpenAI response format: {result}")
            return None
            
        # Ensure score is in valid range
        score = max(0.0, min(1.0, float(result["score"])))
        reason = result.get("reason", "No reason provided")
        
        logger.info(f"E2: OpenAI scored {domain} as {score:.2f} - {reason}")
        return {"score": score, "reason": reason}
        
    except Exception as e:
        logger.error(f"E2: OpenAI evaluation failed for {domain}: {e}")
        return None

def calculate_reliability_score(evidence_item: Dict[str, Any], trusted_sources: Dict[str, List[str]], claim_consensus_map: Dict[str, int]) -> Dict[str, Any]:
    """Calculates the reliability score for a single evidence item using OpenAI (primary) or heuristics (fallback)."""
    
    url = evidence_item.get("url", "")
    claim_text = evidence_item.get("claim_text", "")
    snippet = evidence_item.get("snippet", "")
    
    if not url:
        return {"score": 0.0, "details": ["No URL provided"]}

    domain = get_domain(url)
    
    # Try OpenAI evaluation first
    if openai_client:
        logger.debug(f"E2: Attempting OpenAI evaluation for {domain}")
        openai_result = evaluate_source_reliability_openai(domain, url, snippet, claim_text)
        
        if openai_result:
            return {
                "score": openai_result["score"],
                "details": [f"OpenAI: {openai_result['reason']}"]
            }
        else:
            logger.info(f"E2: OpenAI evaluation failed for {domain}, falling back to heuristics")
    
    # Fallback to heuristic scoring
    logger.debug(f"E2: Using heuristic scoring for {domain}")
    score = 0.5  # Base score
    details = []
    
    # 1. Domain TLD Check
    if domain.endswith(".gov") or domain.endswith(".mil"):
        score += 0.4
        details.append("Government/Military domain (+0.4)")
    elif domain.endswith(".edu"):
        score += 0.3  # Slightly less than gov
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

def _score_single_item(item: Dict[str, Any], trusted_sources: Dict[str, List[str]], claim_consensus_map: Dict[str, int]) -> Dict[str, Any]:
    """
    Score a single evidence item and attach reliability fields.
    Separated for use with ThreadPoolExecutor.
    """
    start = time.time()
    new_item = item.copy()
    result = calculate_reliability_score(new_item, trusted_sources, claim_consensus_map)
    new_item["reliability_score"] = result["score"]
    new_item["reliability_details"] = result["details"]
    new_item["_e2_elapsed"] = time.time() - start
    return new_item

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
    print(f"E2: Scoring {len(evidence_list)} evidence items with up to {MAX_PARALLEL} workers...")
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        future_map = {
            executor.submit(_score_single_item, item, trusted_sources, claim_consensus_map): item
            for item in evidence_list
        }

        for future in as_completed(future_map):
            original_item = future_map[future]
            try:
                new_item = future.result()
                scored_evidence.append(new_item)
                elapsed = new_item.pop("_e2_elapsed", None)
                timing_msg = f" in {elapsed:.2f}s" if elapsed is not None else ""
                print(f"Scored {new_item.get('url', 'N/A')}: {new_item.get('reliability_score', 0.0):.2f}{timing_msg}")
            except Exception as e:
                errored_item = original_item.copy()
                errored_item["reliability_score"] = 0.0
                errored_item["reliability_details"] = [f"Scoring failed: {e}"]
                scored_evidence.append(errored_item)
                print(f"E2: Failed to score {original_item.get('url', 'N/A')}: {e}")

    state["evidence"] = scored_evidence

    dump_node_debug(
        state,
        "E2",
        {"evidence_scored": len(scored_evidence)},
    )
    return state
