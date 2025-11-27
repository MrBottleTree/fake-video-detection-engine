import json
import os
import time
import hashlib
import logging
import uuid
from typing import List, Dict, Any, Optional, TypedDict, Union
from datetime import datetime
import re

import json
import os
import time
import hashlib
import logging
import uuid
from typing import List, Dict, Any, Optional, TypedDict, Union
from datetime import datetime
import re

# Standard Imports
from duckduckgo_search import DDGS
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sentence_transformers import SentenceTransformer, util

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Type Definitions ---

class Claim(TypedDict):
    """Structure of a claim object."""
    id: str
    claim_text: str
    who: Optional[str]
    what: Optional[str]
    when: Optional[str]
    where: Optional[str]

class EvidenceResult(TypedDict):
    """Structure of a single evidence result."""
    url: str
    title: str
    snippet: str
    source: str
    date: Optional[str]
    relevance_score: float

class EvidenceOutput(EvidenceResult):
    """Structure of the final flattened output."""
    claim_id: str
    claim_text: str
    query_variants: List[str]
    retrieval_timestamp: str

# --- Node Implementation ---

def run(state: dict) -> dict:
    """
    E1: Enhanced Web Evidence Retrieval (Robust Version)
    
    Features:
    - Strict typing and validation
    - Robust multi-API fallback (Serper -> Google -> DDG)
    - Granular error handling
    - Minimal user-facing prints, detailed logging
    - Smart queries (Supporting vs Contradicting)
    - Claim ID generation
    - In-memory caching
    """
    print("Node E1: Retrieving Web Evidence...")
    
    # 1. Input Validation
    claims_raw = state.get("claims", [])
    debug = state.get("debug", False)
    use_cache = state.get("use_cache", True)
    
    if debug:
        logger.setLevel(logging.DEBUG)
    
    if not claims_raw:
        print("Warning: No claims found in state. Skipping Web Evidence.")
        logger.warning("E1: 'claims' key missing or empty in state.")
        return state
    
    # Normalize claims to TypedDict with IDs
    claims: List[Claim] = []
    for c in claims_raw:
        claim_id = str(uuid.uuid4()) # Generate new ID by default
        
        if isinstance(c, dict):
            # Preserve existing ID if present
            if "id" in c:
                claim_id = str(c["id"])
                
            claims.append({
                "id": claim_id,
                "claim_text": c.get("claim_text", str(c)), # Fallback if just a dict
                "who": c.get("who"),
                "what": c.get("what"),
                "when": c.get("when"),
                "where": c.get("where")
            })
        else:
             # Handle string claims
             claims.append({
                "id": claim_id,
                "claim_text": str(c),
                "who": None, "what": None, "when": None, "where": None
            })

    evidence_results: List[EvidenceOutput] = []
    searcher = WebSearcher(debug=debug, use_cache=use_cache)
    
    try:
        print(f"Processing {len(claims)} claims...")
        
        for idx, claim in enumerate(claims):
            logger.info(f"Processing claim {idx+1}/{len(claims)}: {claim['claim_text'][:50]}...")
            
            # 2. Smart Query Construction
            query_variants = searcher.construct_queries(claim)
            if not query_variants:
                logger.warning(f"No queries generated for claim: {claim}")
                continue
            
            # 3. Robust Search
            all_results: List[EvidenceResult] = []
            for query in query_variants:
                results = searcher.search_robust(query)
                all_results.extend(results)
                time.sleep(0.2) # Polite rate limiting
            
            # 4. Deduplication & Ranking
            deduplicated = searcher.deduplicate(all_results)
            ranked = searcher.rank_results(deduplicated, claim['claim_text'])
            
            final_results = ranked[:5]
            
            # Flatten results for downstream nodes (E2, E3)
            for res in final_results:
                # Inject claim context into the evidence item
                flat_item = res.copy()
                flat_item["claim_id"] = claim["id"]
                flat_item["claim_text"] = claim["claim_text"]
                flat_item["query_variants"] = query_variants
                flat_item["retrieval_timestamp"] = datetime.now().isoformat()
                
                evidence_results.append(flat_item)
            
        state["evidence"] = evidence_results
        print(f"Node E1: Evidence retrieval complete. Found {len(evidence_results)} total evidence items.")
        
    except Exception as e:
        print(f"Error in E1 node: {e}")
        logger.error(f"Critical error in E1 node: {e}", exc_info=True)
    
    return state


# --- Helper Class ---

class WebSearcher:
    """Robust web searcher with fallbacks and in-memory caching."""
    
    def __init__(self, debug: bool = False, use_cache: bool = True):
        self.debug = debug
        self.use_cache = use_cache
        
        # API Keys
        self.serper_key = os.environ.get("SERPER_API_KEY")
        self.google_key = os.environ.get("GOOGLE_API_KEY")
        self.google_cx = os.environ.get("GOOGLE_CX")
        
        # HTTP Session
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        
        # In-Memory Cache
        self.memory_cache: Dict[str, List[EvidenceResult]] = {}
            
        # Models
        self.embedding_model = None # Lazy load

    def construct_queries(self, claim: Claim) -> List[str]:
        """Generate smart search queries (Supporting & Contradicting)."""
        queries = set()
        text = claim['claim_text']
        
        # 1. Neutral / Fact Check
        queries.add(f"{text} fact check")
        queries.add(f"is it true that {text}")
        
        # 2. Supporting (Positive bias)
        queries.add(f"proof that {text}")
        queries.add(f"evidence for {text}")
        
        # 3. Contradicting (Negative bias - Debunking)
        queries.add(f"{text} debunked")
        queries.add(f"{text} fake")
        queries.add(f"{text} hoax")
        
        # 4. Entity specific (if available)
        if claim['who'] and claim['what']:
            queries.add(f"{claim['who']} {claim['what']} controversy")
            
        return list(queries)

    def search_robust(self, query: str) -> List[EvidenceResult]:
        """Try APIs in order: Cache -> Serper -> Google -> DDG."""
        if not query: return []
        
        # 1. Cache Check
        cached = self._get_from_cache(query)
        if cached:
            logger.debug(f"Cache hit for '{query}'")
            return cached
            
        results = []
        
        # 2. Serper (Primary)
        if self.serper_key:
            try:
                results = self._search_serper(query)
                if results:
                    self._save_to_cache(query, results)
                    return results
            except Exception as e:
                logger.error(f"Serper failed: {e}")
        
        # 3. Google Custom Search (Secondary)
        if self.google_key and self.google_cx:
            try:
                results = self._search_google(query)
                if results:
                    self._save_to_cache(query, results)
                    return results
            except Exception as e:
                logger.error(f"Google CSE failed: {e}")
                
        # 4. DuckDuckGo (Fallback)
        try:
            results = self._search_ddg(query)
            if results:
                self._save_to_cache(query, results)
                return results
        except Exception as e:
            logger.error(f"DDG failed: {e}")
            
        return []

    def _search_serper(self, query: str) -> List[EvidenceResult]:
        url = "https://google.serper.dev/search"
        headers = {'X-API-KEY': self.serper_key, 'Content-Type': 'application/json'}
        payload = json.dumps({"q": query, "num": 5})
        
        resp = self.session.post(url, headers=headers, data=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        results: List[EvidenceResult] = []
        for item in data.get("organic", []):
            results.append({
                "url": item.get("link", ""),
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "source": "serper",
                "date": item.get("date"),
                "relevance_score": 0.0 # Calc later
            })
        return results

    def _search_google(self, query: str) -> List[EvidenceResult]:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": self.google_key, "cx": self.google_cx, "q": query, "num": 5}
        
        resp = self.session.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        results: List[EvidenceResult] = []
        for item in data.get("items", []):
            results.append({
                "url": item.get("link", ""),
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "source": "google",
                "date": None,
                "relevance_score": 0.0
            })
        return results

    def _search_ddg(self, query: str) -> List[EvidenceResult]:
        results: List[EvidenceResult] = []
        with DDGS() as ddgs:
            # DDGS can be unstable, wrap generator
            ddg_gen = ddgs.text(query, max_results=5)
            if ddg_gen:
                for r in ddg_gen:
                    results.append({
                        "url": r.get("href", ""),
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "source": "ddg",
                        "date": None,
                        "relevance_score": 0.0
                    })
        return results

    def deduplicate(self, results: List[EvidenceResult]) -> List[EvidenceResult]:
        seen_urls = set()
        unique = []
        for r in results:
            if r['url'] not in seen_urls:
                seen_urls.add(r['url'])
                unique.append(r)
        return unique

    def rank_results(self, results: List[EvidenceResult], query_text: str) -> List[EvidenceResult]:
        if not results: return []
        
        # Lazy load model
        if not self.embedding_model:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")
                return results # Return unranked
        
        try:
            query_emb = self.embedding_model.encode(query_text, convert_to_tensor=True)
            
            for r in results:
                text = f"{r['title']} {r['snippet']}"
                doc_emb = self.embedding_model.encode(text, convert_to_tensor=True)
                score = util.cos_sim(query_emb, doc_emb).item()
                r['relevance_score'] = float(score)
                
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
        except Exception as e:
            logger.error(f"Ranking failed: {e}")
            
        return results

    def _get_from_cache(self, query: str) -> Optional[List[EvidenceResult]]:
        if not self.use_cache: return None
        return self.memory_cache.get(query)

    def _save_to_cache(self, query: str, results: List[EvidenceResult]):
        if not self.use_cache: return
        self.memory_cache[query] = results
