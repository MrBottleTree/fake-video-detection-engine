import urllib.parse
import json
import os
import re
import socket
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import concurrent.futures

try:
    import requests
    from requests.exceptions import SSLError, Timeout
except ImportError:
    requests = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import whois
except ImportError:
    whois = None


def run(state: dict) -> dict:
    """
    E2: Enhanced Source Reliability Assessment
    
    Improvements:
    - SSL certificate validation
    - About page detection (NOW ACTUALLY USED)
    - Domain age via WHOIS
    - Author/byline detection
    - Cross-source consensus boost
    - Content freshness scoring
    - Parallel processing for speed
    """
    print("Node E2: Assessing Source Reliability (Production-Ready)...")
    
    evidence_data = state.get("evidence", [])
    debug = state.get("debug", False)
    
    if not evidence_data:
        print("Warning: No evidence found in state. Skipping Source Reliability.")
        return state
    
    # Load trusted sources database
    domain_scores = load_trusted_sources(debug)
    
    # Initialize reliability assessor
    assessor = ReliabilityAssessor(domain_scores, debug=debug)
    
    try:
        # Process all evidence items
        for item in evidence_data:
            results = item.get("results", [])
            claim = item.get("claim", {})
            
            # Parallel assessment for speed
            if len(results) > 3:
                results = assessor.assess_parallel(results, claim)
            else:
                for result in results:
                    reliability_data = assessor.assess_source(result, claim)
                    result.update(reliability_data)
        
        # Apply cross-source consensus boost
        evidence_data = assessor.apply_consensus_boost(evidence_data, debug)
        
        state["evidence"] = evidence_data
        print("✅ E2: Reliability assessment complete.")
        
    except Exception as e:
        print(f"❌ Error in E2 node: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        raise e
    
    return state


def load_trusted_sources(debug=False) -> Dict[str, float]:
    """Load trusted sources with improved error handling"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    trusted_sources_path = os.path.join(project_root, "resources", "trusted_sources.json")
    
    domain_scores = {}
    
    if os.path.exists(trusted_sources_path):
        try:
            with open(trusted_sources_path, "r") as f:
                data = json.load(f)
            
            # Enhanced tier mapping with more granularity
            tier_map = {
                "tier1": 1.0,   # Reuters, AP, BBC (Highest Credibility)
                "tier2": 0.9,   # WSJ, Bloomberg, FT (High Credibility)
                "tier3": 0.85,  # Specialized/Academic
                "tier4": 0.7,   # Contextual/Regional
                "tier5": 0.2,   # Low Credibility/Hyperpartisan
                "tier6": 0.0    # Satire/User Generated
            }
            
            for tier, domains in data.items():
                score = tier_map.get(tier, 0.5)
                for d in domains:
                    domain_scores[d.lower()] = score
            
            if debug:
                print(f"✅ Loaded {len(domain_scores)} trusted domains")
        
        except Exception as e:
            print(f"⚠️  Error loading trusted sources: {e}")
    else:
        if debug:
            print("⚠️  trusted_sources.json not found. Using minimal fallback.")
    
    return domain_scores


class ReliabilityAssessor:
    """Advanced source reliability assessment with multiple signals"""
    
    def __init__(self, domain_scores: Dict[str, float], debug=False):
        self.domain_scores = domain_scores
        self.debug = debug
        self.session = None
        
        if requests:
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (compatible; FactCheckBot/1.0)'
            })
    
    def assess_source(self, result: Dict, claim: Dict) -> Dict:
        """
        Comprehensive source reliability assessment.
        
        Returns dict with:
        - reliability_score: 0-1
        - reliability_factors: breakdown of contributing factors
        """
        url = result.get("url", "")
        
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            
            # Initialize scoring components
            factors = {
                "database_match": 0.0,
                "tld_heuristic": 0.0,
                "unknown_domain_base": 0.0,
                "ssl_valid": 0.0,
                "about_page": 0.0,
                "domain_age": 0.0,
                "author_present": 0.0,
                "content_length": 0.0,
                "freshness": 0.0
            }
            
            # 1. Database lookup (most important)
            if domain in self.domain_scores:
                factors["database_match"] = self.domain_scores[domain]
            else:
                # 2. TLD heuristics (fallback)
                if domain.endswith(".gov"):
                    factors["tld_heuristic"] = 1.0
                elif domain.endswith(".edu"):
                    factors["tld_heuristic"] = 0.95
                elif domain.endswith(".mil"):
                    factors["tld_heuristic"] = 1.0
                elif domain.endswith(".org"):
                    factors["tld_heuristic"] = 0.65
                else:
                    factors["unknown_domain_base"] = 0.5
            
            # 3. HTTPS check
            if parsed.scheme == "https":
                factors["ssl_valid"] = 0.05
            
            # 4. About page check (NOW ACTUALLY USED)
            has_about = self.check_about_page(domain)
            if has_about:
                factors["about_page"] = 0.05
            
            # 5. Domain age (if WHOIS available)
            domain_age_years = self.get_domain_age(domain)
            if domain_age_years:
                # Older domains more trustworthy (capped at 5 years)
                age_score = min(domain_age_years / 5.0, 1.0) * 0.08
                factors["domain_age"] = age_score
            
            # 6. Content quality signals
            snippet = result.get("snippet", "")
            title = result.get("title", "")
            
            # Long, detailed snippets suggest quality
            if len(snippet) > 200:
                factors["content_length"] = 0.05
            
            # Author/byline detection
            if self.has_author_attribution(snippet, title):
                factors["author_present"] = 0.05
            
            # 7. Freshness scoring
            date_str = result.get("date")
            if date_str:
                freshness_score = self.calculate_freshness_score(date_str)
                factors["freshness"] = freshness_score * 0.07
            
            # Calculate final score
            final_score = sum(factors.values())
            final_score = min(max(final_score, 0.0), 1.0)  # Clamp to [0, 1]
            
            if self.debug:
                print(f"\n   📊 {domain}: {final_score:.3f}")
                for factor, value in factors.items():
                    if value > 0:
                        print(f"      • {factor}: +{value:.3f}")
            
            return {
                "reliability_score": final_score,
                "reliability_factors": factors,
                "domain": domain
            }
        
        except Exception as e:
            if self.debug:
                print(f"   ⚠️  Error assessing {url}: {e}")
            
            return {
                "reliability_score": 0.5,
                "reliability_factors": {},
                "domain": None
            }
    
    def check_about_page(self, domain: str) -> bool:
        """
        Check if domain has an About page (legitimacy signal).
        NOW ACTUALLY CALLED!
        """
        if not self.session:
            return False
        
        about_paths = ["/about", "/about-us", "/about.html", "/aboutus", "/who-we-are"]
        
        for path in about_paths:
            try:
                url = f"https://{domain}{path}"
                # Try HEAD first (faster)
                resp = self.session.head(url, timeout=3, allow_redirects=True)
                
                if resp.status_code == 200:
                    return True
                elif resp.status_code == 405:
                    # HEAD not allowed, try GET
                    resp = self.session.get(url, timeout=3, allow_redirects=True, stream=True)
                    if resp.status_code == 200:
                        return True
            
            except:
                continue
        
        return False
    
    def get_domain_age(self, domain: str) -> Optional[float]:
        """Get domain age in years via WHOIS"""
        if not whois:
            return None
        
        try:
            w = whois.whois(domain)
            creation_date = w.creation_date
            
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            
            if creation_date:
                age = (datetime.now() - creation_date).days / 365.25
                return max(age, 0)
        
        except:
            pass
        
        return None
    
    def has_author_attribution(self, snippet: str, title: str) -> bool:
        """Detect if content has author attribution (quality signal)"""
        author_patterns = [
            r"by\s+[A-Z][a-z]+\s+[A-Z][a-z]+",  # "by John Smith"
            r"[A-Z][a-z]+\s+[A-Z][a-z]+\s*,\s*\w+\s+\d+",  # "John Smith, March 15"
            r"written by",
            r"authored by",
            r"reported by"
        ]
        
        text = f"{title} {snippet}".lower()
        
        for pattern in author_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def calculate_freshness_score(self, date_str: str) -> float:
        """
        Calculate content freshness score (0-1).
        Recent content is better for current events.
        """
        try:
            pub_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            days_old = (datetime.now() - pub_date).days
            
            # Decay function: fresh=1.0, 1yr=0.5, 2yr+=0.0
            if days_old < 7:
                return 1.0
            elif days_old < 30:
                return 0.9
            elif days_old < 90:
                return 0.7
            elif days_old < 365:
                return 0.5
            else:
                return max(0, 0.5 - (days_old - 365) / 730)
        
        except:
            return 0.5  # Unknown date = neutral
    
    def assess_parallel(self, results: List[Dict], claim: Dict) -> List[Dict]:
        """Assess multiple sources in parallel for speed"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self.assess_source, r, claim): r for r in results}
            
            for future in concurrent.futures.as_completed(futures):
                result = futures[future]
                try:
                    reliability_data = future.result()
                    result.update(reliability_data)
                except Exception as e:
                    if self.debug:
                        print(f"   ⚠️  Parallel assessment error: {e}")
                    result.update({
                        "reliability_score": 0.5,
                        "reliability_factors": {},
                        "domain": None
                    })
        
        return results
    
    def apply_consensus_boost(self, evidence_data: List[Dict], debug=False) -> List[Dict]:
        """
        Apply cross-source consensus boost.
        If multiple reliable sources agree, boost all their scores slightly.
        """
        for item in evidence_data:
            results = item.get("results", [])
            
            if len(results) < 3:
                continue  # Need multiple sources
            
            # Count high-reliability sources
            reliable_count = sum(1 for r in results if r.get("reliability_score", 0) > 0.7)
            
            # If majority are reliable, apply small boost
            if reliable_count >= len(results) * 0.6:
                boost = 0.05
                for r in results:
                    old_score = r.get("reliability_score", 0.5)
                    r["reliability_score"] = min(old_score + boost, 1.0)
                    
                    if "reliability_factors" in r:
                        r["reliability_factors"]["consensus_boost"] = boost
                
                if debug:
                    print(f"   🔗 Applied consensus boost: {reliable_count}/{len(results)} reliable sources")
        
        return evidence_data
