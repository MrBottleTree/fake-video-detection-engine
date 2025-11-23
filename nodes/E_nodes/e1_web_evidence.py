import json
import os
import time
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import re

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    requests = None

try:
    from sentence_transformers import SentenceTransformer, util
    EMBEDDING_MODEL = None  # Lazy load
except ImportError:
    SentenceTransformer = None

try:
    import redis
    REDIS_CLIENT = None  # Lazy load
except ImportError:
    redis = None


def run(state: dict) -> dict:
    """
    E1: Enhanced Web Evidence Retrieval
    
    Features:
    - Smart query construction with entity emphasis
    - Multi-API fallback with exponential backoff
    - Result deduplication and semantic clustering
    - Temporal filtering preferences (recent > old)
    - Caching layer for repeated claims
    - Robust error handling and rate limiting
    """
    print("Node E1: Retrieving Web Evidence (Production-Ready)...")
    
    claims = state.get("claims", [])
    debug = state.get("debug", False)
    use_cache = state.get("use_cache", True)
    
    if not claims:
        print("Warning: No claims found in state. Skipping Web Evidence.")
        return state
    
    evidence_results = []
    searcher = EnhancedWebSearcher(debug=debug, use_cache=use_cache)
    
    try:
        for idx, claim in enumerate(claims):
            if debug:
                print(f"\n{'='*60}")
                print(f"Processing Claim {idx+1}/{len(claims)}")
                print(f"{'='*60}")
            
            # Construct optimized query
            query_variants = searcher.construct_smart_queries(claim)
            
            if not query_variants:
                if debug:
                    print(f"Warning: No valid query constructed for claim: {claim}")
                continue
            
            # Execute searches with all query variants
            all_results = []
            for query in query_variants:
                if debug:
                    print(f"\n📝 Query: '{query}'")
                
                results = searcher.search(query, k=7)  # Fetch more, filter later
                all_results.extend(results)
                
                # Rate limiting
                time.sleep(0.5)
            
            # Deduplicate and rank results
            deduplicated = searcher.deduplicate_results(all_results)
            ranked = searcher.rank_by_relevance(deduplicated, claim)
            
            # Keep top 5 most relevant
            final_results = ranked[:5]
            
            if debug:
                print(f"\n✅ Retrieved {len(final_results)} unique results")
                for i, r in enumerate(final_results[:3], 1):
                    print(f"   {i}. {r['title'][:60]}...")
            
            evidence_results.append({
                "claim": claim,
                "query_variants": query_variants,
                "results": final_results,
                "timestamp": datetime.now().isoformat()
            })
        
        print(f"\n✅ E1: Retrieved evidence for {len(evidence_results)} claims.")
        state["evidence"] = evidence_results
        
    except Exception as e:
        print(f"❌ Error in E1 node: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        raise e
    
    return state


class EnhancedWebSearcher:
    """Production-ready web searcher with multiple backends and intelligent query construction"""
    
    def __init__(self, debug=False, use_cache=True):
        self.debug = debug
        self.use_cache = use_cache
        
        # API Keys
        self.serper_api_key = os.environ.get("SERPER_API_KEY")
        self.google_api_key = os.environ.get("GOOGLE_API_KEY")
        self.google_cx = os.environ.get("GOOGLE_CX")
        
        # Initialize search backends
        self.ddgs = DDGS() if DDGS else None
        
        # Session with retry strategy
        if requests:
            self.session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET", "POST"]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
        else:
            self.session = None
        
        # Cache layer
        if redis and use_cache:
            try:
                self.cache = redis.Redis(
                    host=os.environ.get('REDIS_HOST', 'localhost'),
                    port=int(os.environ.get('REDIS_PORT', 6379)),
                    db=0,
                    decode_responses=True,
                    socket_connect_timeout=2
                )
                self.cache.ping()  # Test connection
                if self.debug:
                    print("✅ Redis cache connected")
            except Exception as e:
                self.cache = None
                if self.debug:
                    print(f"⚠️  Redis cache unavailable ({e}), using in-memory")
                self.memory_cache = {}
        else:
            self.cache = None
            self.memory_cache = {} if use_cache else None
        
        # Load sentence transformer for semantic ranking (lazy)
        self.embedding_model = None
    
    def construct_smart_queries(self, claim: Dict[str, Any]) -> List[str]:
        """
        Construct multiple query variants optimized for different search engines.
        """
        queries = []
        
        if isinstance(claim, dict):
            who = claim.get("who", "").strip()
            what = claim.get("what", "").strip()
            when = claim.get("when", "").strip()
            where = claim.get("where", "").strip()
            
            # Query 1: Full structured query (most specific)
            parts = [p for p in [who, what, where, when] if p]
            if parts:
                full_query = " ".join(parts)
                queries.append(full_query[:120])  # Limit length
            
            # Query 2: Entity-focused (who + what, most important)
            if who and what:
                entity_query = f'"{who}" {what}'
                if when:
                    entity_query += f' {when}'
                queries.append(entity_query[:120])
            
            # Query 3: What-focused with context
            if what:
                context_query = what
                if when:
                    context_query += f' {when}'
                if where:
                    context_query += f' {where}'
                queries.append(context_query[:120])
            
            # Query 4: Verification-oriented
            if who and what:
                verify_query = f'{who} {what} fact check'
                queries.append(verify_query[:120])
        
        elif isinstance(claim, (list, tuple)):
            claim_text = " ".join([str(c) for c in claim if c])
            queries.append(claim_text[:120])
        
        else:
            claim_text = str(claim).strip()
            if claim_text:
                queries.append(claim_text[:120])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            q_clean = q.lower().strip()
            if q_clean and q_clean not in seen:
                seen.add(q_clean)
                unique_queries.append(q)
        
        return unique_queries
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        return f"search:{hashlib.md5(query.encode()).hexdigest()}"
    
    def _get_from_cache(self, query: str) -> Optional[List[Dict]]:
        """Retrieve results from cache"""
        if not self.use_cache:
            return None
        
        cache_key = self._get_cache_key(query)
        
        # Try Redis
        if self.cache:
            try:
                cached = self.cache.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                if self.debug:
                    print(f"Cache read error: {e}")
        
        # Fallback to memory cache
        if self.memory_cache is not None:
            return self.memory_cache.get(cache_key)
        
        return None
    
    def _save_to_cache(self, query: str, results: List[Dict], ttl: int = 3600):
        """Save results to cache with TTL"""
        if not self.use_cache:
            return
        
        cache_key = self._get_cache_key(query)
        
        # Try Redis with TTL
        if self.cache:
            try:
                self.cache.setex(cache_key, ttl, json.dumps(results))
                return
            except Exception as e:
                if self.debug:
                    print(f"Cache write error: {e}")
        
        # Fallback to memory cache (no TTL)
        if self.memory_cache is not None:
            self.memory_cache[cache_key] = results
    
    def search(self, query: str, k: int = 7) -> List[Dict[str, Any]]:
        """
        Execute search with multi-API fallback and caching.
        """
        if not query or not query.strip():
            return []
        
        query = query.strip()
        
        # Check cache
        cached_results = self._get_from_cache(query)
        if cached_results:
            if self.debug:
                print(f"   💾 Cache hit for '{query}'")
            return cached_results
        
        results = []
        
        # 1. Try Serper API (Best quality)
        if self.serper_api_key and self.session:
            results = self._search_serper(query, k)
            if results:
                self._save_to_cache(query, results)
                return results
        
        # 2. Try Google Custom Search API
        if self.google_api_key and self.google_cx and self.session:
            results = self._search_google_custom(query, k)
            if results:
                self._save_to_cache(query, results)
                return results
        
        # 3. Fallback to DuckDuckGo (Free)
        if self.ddgs:
            results = self._search_duckduckgo(query, k)
            if results:
                self._save_to_cache(query, results)
                return results
        
        # No results from any API
        if self.debug:
            print(f"   ⚠️  No results from any search API for '{query}'")
        
        return []
    
    def _search_serper(self, query: str, k: int) -> List[Dict]:
        """Search using Serper.dev API (Google wrapper)"""
        if self.debug:
            print(f"   🔍 Trying Serper API...")
        
        try:
            url = "https://google.serper.dev/search"
            payload = json.dumps({
                "q": query,
                "num": k,
                "gl": "us",  # Geographic location
                "hl": "en"   # Language
            })
            headers = {
                'X-API-KEY': self.serper_api_key,
                'Content-Type': 'application/json'
            }
            
            response = self.session.post(url, headers=headers, data=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                organic = data.get("organic", [])
                results = []
                
                for item in organic:
                    # Extract date if available
                    date_str = item.get("date", "")
                    pub_date = self._parse_date(date_str)
                    
                    results.append({
                        "url": item.get("link"),
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "source": "serper",
                        "position": item.get("position", 99),
                        "date": pub_date
                    })
                
                if self.debug:
                    print(f"   ✅ Serper: {len(results)} results")
                return results
            
            else:
                if self.debug:
                    print(f"   ⚠️  Serper API error: {response.status_code}")
        
        except Exception as e:
            if self.debug:
                print(f"   ⚠️  Serper API failed: {e}")
        
        return []
    
    def _search_google_custom(self, query: str, k: int) -> List[Dict]:
        """Search using Google Custom Search API"""
        if self.debug:
            print(f"   🔍 Trying Google Custom Search API...")
        
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_api_key,
                "cx": self.google_cx,
                "q": query,
                "num": min(k, 10)  # Max 10 per request
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])
                results = []
                
                for item in items:
                    results.append({
                        "url": item.get("link"),
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "source": "google_custom",
                        "date": None  # Google CSE doesn't provide dates easily
                    })
                
                if self.debug:
                    print(f"   ✅ Google CSE: {len(results)} results")
                return results
            
            else:
                if self.debug:
                    print(f"   ⚠️  Google CSE error: {response.status_code}")
        
        except Exception as e:
            if self.debug:
                print(f"   ⚠️  Google CSE failed: {e}")
        
        return []
    
    def _search_duckduckgo(self, query: str, k: int) -> List[Dict]:
        """Search using DuckDuckGo (Free, rate limited)"""
        if self.debug:
            print(f"   🔍 Trying DuckDuckGo...")
        
        try:
            ddg_gen = self.ddgs.text(query, max_results=k)
            results = []
            
            for r in ddg_gen:
                results.append({
                    "url": r.get("href"),
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "source": "duckduckgo",
                    "date": None  # DDG doesn't provide dates
                })
            
            if self.debug:
                print(f"   ✅ DuckDuckGo: {len(results)} results")
            return results
        
        except Exception as e:
            if self.debug:
                print(f"   ⚠️  DuckDuckGo failed: {e}")
            # DDG rate limiting - wait before retry
            time.sleep(2)
        
        return []
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse various date formats to ISO format"""
        if not date_str:
            return None
        
        # Common patterns
        patterns = [
            r'(\d{4})-(\d{2})-(\d{2})',  # 2024-11-23
            r'(\d{1,2})\s+(hours?|days?|weeks?|months?)\s+ago',  # "2 hours ago"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, date_str, re.IGNORECASE)
            if match:
                if '-' in date_str:
                    return date_str  # Already ISO
                else:
                    # Relative date
                    return datetime.now().isoformat()
        
        return None
    
    def deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """
        Remove duplicate URLs and near-duplicate content.
        """
        seen_urls = set()
        seen_content = []
        unique_results = []
        
        for result in results:
            url = result.get("url", "")
            title = result.get("title", "").lower()
            snippet = result.get("snippet", "").lower()
            
            # Skip if URL already seen
            if url in seen_urls:
                continue
            
            # Check content similarity
            is_duplicate = False
            for prev_title, prev_snippet in seen_content:
                # Simple similarity: check overlap
                title_overlap = len(set(title.split()) & set(prev_title.split()))
                snippet_overlap = len(set(snippet.split()) & set(prev_snippet.split()))
                
                # Consider duplicate if high overlap
                if title_overlap > len(title.split()) * 0.7 or snippet_overlap > len(snippet.split()) * 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_urls.add(url)
                seen_content.append((title, snippet))
                unique_results.append(result)
        
        return unique_results
    
    def rank_by_relevance(self, results: List[Dict], claim: Dict) -> List[Dict]:
        """
        Rank results by relevance to claim.
        """
        if not results:
            return []
        
        # Extract claim text
        if isinstance(claim, dict):
            claim_text = " ".join([str(v) for v in claim.values() if v])
        else:
            claim_text = str(claim)
        
        # Load embedding model if available (lazy)
        if SentenceTransformer and not self.embedding_model:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                if self.debug:
                    print("   🤖 Loaded sentence transformer for semantic ranking")
            except:
                pass
        
        # Calculate scores
        scored_results = []
        for result in results:
            score = 0.0
            
            # 1. Position score (0-1, higher is better)
            position = result.get("position", 50)
            position_score = max(0, 1 - (position / 50))
            score += position_score * 0.3
            
            # 2. Recency score (0-1, higher is better)
            date_str = result.get("date")
            if date_str:
                try:
                    pub_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    days_old = (datetime.now() - pub_date).days
                    recency_score = max(0, 1 - (days_old / 365))
                    score += recency_score * 0.2
                except:
                    pass
            
            # 3. Semantic similarity (0-1, higher is better)
            if self.embedding_model:
                try:
                    result_text = f"{result.get('title', '')} {result.get('snippet', '')}"
                    claim_emb = self.embedding_model.encode(claim_text, convert_to_tensor=True)
                    result_emb = self.embedding_model.encode(result_text, convert_to_tensor=True)
                    similarity = util.cos_sim(claim_emb, result_emb).item()
                    score += similarity * 0.5
                except:
                    pass
            
            result["relevance_score"] = score
            scored_results.append(result)
        
        # Sort by score (descending)
        scored_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return scored_results
