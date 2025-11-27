import logging
import os
import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from duckduckgo_search import DDGS
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sentence_transformers import SentenceTransformer, CrossEncoder, util

from nodes.utils.schema import normalize_claims

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- Type Definitions ---

class Claim(TypedDict, total=False):
    id: str
    claim_text: str
    who: Optional[str]
    what: Optional[str]
    when: Optional[str]
    where: Optional[str]
    source: Optional[str]
    confidence: Optional[float]


class EvidenceResult(TypedDict, total=False):
    url: str
    title: str
    snippet: str
    source: str
    date: Optional[str]
    relevance_score: float
    query: str
    stance: str
    ranking_mode: str


class EvidenceItem(TypedDict, total=False):
    claim_id: str
    claim_text: str
    claim_source: Optional[str]
    claim_confidence: Optional[float]
    url: str
    title: str
    snippet: str
    source: str
    date: Optional[str]
    relevance_score: float
    rank: int
    query: str
    retrieved_at: str
    stance: str
    ranking_mode: str


# --- Node Implementation ---

def run(state: dict) -> dict:
    """
    E1: Enhanced Web Evidence Retrieval

    - Robust search with multi-API fallback (Serper -> Google -> DDG)
    - Embedding-based ranking + optional cross-encoder rerank
    - Per-domain caps to avoid single-source domination
    - NLI-backed stance (support/contradict) with heuristic fallback
    - Outputs a flat evidence list annotated with claim metadata
    """
    print("Node E1: Retrieving Web Evidence...")

    claims_raw = state.get("claims", [])
    debug = state.get("debug", False)
    use_cache = state.get("use_cache", True)
    search_depth = max(1, int(state.get("search_depth", 3)))
    support_top_n = int(state.get("support_top_n", 3))
    contradict_top_n = int(state.get("contradict_top_n", 3))
    per_domain_cap = max(1, int(state.get("per_domain_cap", 2)))
    use_nli_stance = bool(state.get("use_nli_stance", True))
    rerank_top_k = int(state.get("rerank_top_k", 10))

    if debug:
        logger.setLevel(logging.DEBUG)

    if not claims_raw:
        print("Warning: No claims found in state. Skipping Web Evidence.")
        logger.warning("E1: 'claims' key missing or empty in state.")
        return state

    claims: List[Claim] = normalize_claims(claims_raw)  # type: ignore
    state["claims"] = claims

    evidence_items: List[EvidenceItem] = []
    searcher = WebSearcher(
        debug=debug,
        use_cache=use_cache,
        search_depth=search_depth,
        per_domain_cap=per_domain_cap,
        use_nli_stance=use_nli_stance,
        rerank_top_k=rerank_top_k,
    )

    try:
        print(f"Processing {len(claims)} claims...")

        for idx, claim in enumerate(claims):
            logger.info("Processing claim %s/%s: %s", idx + 1, len(claims), claim["claim_text"][:80])

            query_variants = searcher.construct_queries(claim)
            if not query_variants:
                logger.warning("No queries generated for claim: %s", claim)
                continue

            all_results: List[EvidenceResult] = []
            for query in query_variants:
                results = searcher.search_robust(query)
                for r in results:
                    r["query"] = query
                all_results.extend(results)
                time.sleep(0.2)  # Polite rate limiting

            deduplicated = searcher.deduplicate(all_results)
            ranked = searcher.rank_results(deduplicated, claim["claim_text"])
            reranked = searcher.rerank_cross_encoder(ranked, claim["claim_text"])
            capped = searcher.cap_per_domain(reranked)
            with_stance = searcher.attach_stance(capped, claim["claim_text"])

            support_results = [r for r in with_stance if r.get("stance") == "support"]
            contradict_results = [r for r in with_stance if r.get("stance") == "contradict"]
            final_results = support_results[:support_top_n] + contradict_results[:contradict_top_n]

            for rank_idx, res in enumerate(final_results, start=1):
                evidence_items.append(
                    {
                        "claim_id": claim["id"],
                        "claim_text": claim["claim_text"],
                        "claim_source": claim.get("source"),
                        "claim_confidence": claim.get("confidence"),
                        "url": res.get("url", ""),
                        "title": res.get("title", ""),
                        "snippet": res.get("snippet", ""),
                        "source": res.get("source", ""),
                        "date": res.get("date"),
                        "relevance_score": res.get("relevance_score", 0.0),
                        "rank": rank_idx,
                        "query": res.get("query", ""),
                        "retrieved_at": datetime.now().isoformat(),
                        "stance": res.get("stance", "support"),
                        "ranking_mode": res.get("ranking_mode", "semantic"),
                    }
                )

        state["evidence"] = evidence_items
        print(f"Node E1: Evidence retrieval complete. Collected {len(evidence_items)} items.")

    except Exception as e:
        print(f"Error in E1 node: {e}")
        logger.error("Critical error in E1 node: %s", e, exc_info=True)

    return state


# --- Helper Class ---

class WebSearcher:
    """Robust web searcher with fallbacks, ranking, and stance detection."""

    def __init__(
        self,
        debug: bool = False,
        use_cache: bool = True,
        search_depth: int = 3,
        per_domain_cap: int = 2,
        use_nli_stance: bool = True,
        rerank_top_k: int = 10,
    ):
        self.debug = debug
        self.use_cache = use_cache
        self.search_depth = max(1, search_depth)
        self.per_domain_cap = max(1, per_domain_cap)
        self.use_nli_stance = use_nli_stance
        self.rerank_top_k = max(1, rerank_top_k)

        self.serper_key = os.environ.get("SERPER_API_KEY")
        self.google_key = os.environ.get("GOOGLE_API_KEY")
        self.google_cx = os.environ.get("GOOGLE_CX")

        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

        self.memory_cache: Dict[str, List[EvidenceResult]] = {}

        self.embedding_model: Optional[SentenceTransformer] = None
        self.cross_encoder: Optional[CrossEncoder] = None
        self.nli_model: Optional[CrossEncoder] = None
        self.embedding_loaded = False
        self.cross_loaded = False
        self.nli_loaded = False

    # --- Query construction ---
    def construct_queries(self, claim: Claim) -> List[str]:
        queries = set()
        text = claim.get("claim_text") or ""

        if not text:
            return []

        queries.add(f"{text} fact check")
        queries.add(f"is it true that {text}")

        queries.add(f"proof that {text}")
        queries.add(f"evidence for {text}")

        queries.add(f"{text} debunked")
        queries.add(f"{text} fake")
        queries.add(f"{text} hoax")

        if claim.get("who") and claim.get("what"):
            queries.add(f"{claim['who']} {claim['what']} controversy")

        return list(queries)

    # --- Search orchestration ---
    def search_robust(self, query: str) -> List[EvidenceResult]:
        if not query:
            return []

        cached = self._get_from_cache(query)
        if cached:
            logger.debug("Cache hit for '%s'", query)
            return cached

        results: List[EvidenceResult] = []

        if self.serper_key:
            try:
                results = self._search_serper(query)
                if results:
                    self._save_to_cache(query, results)
                    return results
            except Exception as e:
                logger.error("Serper failed: %s", e)

        if self.google_key and self.google_cx:
            try:
                results = self._search_google(query)
                if results:
                    self._save_to_cache(query, results)
                    return results
            except Exception as e:
                logger.error("Google CSE failed: %s", e)

        try:
            results = self._search_ddg(query)
            if results:
                self._save_to_cache(query, results)
                return results
        except Exception as e:
            logger.error("DDG failed: %s", e)

        return []

    def _search_serper(self, query: str) -> List[EvidenceResult]:
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": self.serper_key, "Content-Type": "application/json"}
        payload = {"q": query, "num": 5 * self.search_depth}

        resp = self.session.post(url, headers=headers, json=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        results: List[EvidenceResult] = []
        for item in data.get("organic", []):
            results.append(
                {
                    "url": item.get("link", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "serper",
                    "date": item.get("date"),
                    "relevance_score": 0.0,
                }
            )
        return results

    def _search_google(self, query: str) -> List[EvidenceResult]:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": self.google_key, "cx": self.google_cx, "q": query, "num": 5 * self.search_depth}

        resp = self.session.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        results: List[EvidenceResult] = []
        for item in data.get("items", []):
            results.append(
                {
                    "url": item.get("link", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "google",
                    "date": None,
                    "relevance_score": 0.0,
                }
            )
        return results

    def _search_ddg(self, query: str) -> List[EvidenceResult]:
        results: List[EvidenceResult] = []
        with DDGS() as ddgs:
            ddg_gen = ddgs.text(query, max_results=5 * self.search_depth)
            if ddg_gen:
                for r in ddg_gen:
                    results.append(
                        {
                            "url": r.get("href", ""),
                            "title": r.get("title", ""),
                            "snippet": r.get("body", ""),
                            "source": "ddg",
                            "date": None,
                            "relevance_score": 0.0,
                        }
                    )
        return results

    # --- Deduplication & Ranking ---
    def deduplicate(self, results: List[EvidenceResult]) -> List[EvidenceResult]:
        seen = set()
        unique: List[EvidenceResult] = []
        for r in results:
            url = r.get("url", "") or ""
            title = r.get("title", "") or ""
            snippet = r.get("snippet", "") or ""
            key = url or f"{title}_{snippet}"
            if key in seen:
                continue
            seen.add(key)
            unique.append(r)

        filtered: List[EvidenceResult] = []
        for candidate in unique:
            text = f"{candidate.get('title','')} {candidate.get('snippet','')}"
            tokens = set(text.lower().split())
            is_dup = False
            for kept in filtered:
                kept_text = f"{kept.get('title','')} {kept.get('snippet','')}"
                kept_tokens = set(kept_text.lower().split())
                if not tokens or not kept_tokens:
                    continue
                overlap = len(tokens & kept_tokens) / max(1, len(tokens | kept_tokens))
                if overlap > 0.7:
                    is_dup = True
                    break
            if not is_dup:
                filtered.append(candidate)
        return filtered

    def rank_results(self, results: List[EvidenceResult], query_text: str) -> List[EvidenceResult]:
        if not results:
            return []

        ranking_mode = "semantic"
        if not self.embedding_model and not self.embedding_loaded:
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.embedding_loaded = True
            except Exception as e:
                logger.warning("Could not load embedding model: %s", e)
                self.embedding_loaded = True
                self.embedding_model = None

        if self.embedding_model:
            try:
                query_emb = self.embedding_model.encode(query_text, convert_to_tensor=True)
                for r in results:
                    text = f"{r.get('title','')} {r.get('snippet','')}"
                    doc_emb = self.embedding_model.encode(text, convert_to_tensor=True)
                    score = util.cos_sim(query_emb, doc_emb).item()
                    r["relevance_score"] = float(score)
                results.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
                for r in results:
                    r["ranking_mode"] = ranking_mode
                return results
            except Exception as e:
                logger.error("Ranking failed, falling back: %s", e)

        ranking_mode = "fallback"
        query_tokens = set(query_text.lower().split())
        scored = []
        for idx, r in enumerate(results):
            text_tokens = set(f"{r.get('title','')} {r.get('snippet','')}".lower().split())
            overlap = len(query_tokens & text_tokens) / max(1, len(query_tokens))
            position_score = max(0.0, 1 - (idx / 10))
            relevance = 0.6 * overlap + 0.4 * position_score
            r["relevance_score"] = float(relevance)
            r["ranking_mode"] = ranking_mode
            scored.append(r)
        scored.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        return scored

    def rerank_cross_encoder(self, results: List[EvidenceResult], query_text: str) -> List[EvidenceResult]:
        if not results:
            return []
        if not self.cross_encoder and not self.cross_loaded:
            try:
                self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                self.cross_loaded = True
            except Exception as e:
                logger.warning("Cross-encoder load failed: %s", e)
                self.cross_loaded = True
                self.cross_encoder = None
        if not self.cross_encoder:
            return results

        top_candidates = results[: min(self.rerank_top_k, len(results))]
        pairs = [(query_text, f"{r.get('title','')} {r.get('snippet','')}") for r in top_candidates]
        try:
            scores = self.cross_encoder.predict(pairs)
            for r, s in zip(top_candidates, scores):
                r["relevance_score"] = float(s)
                r["ranking_mode"] = "cross_encoder"
            top_candidates.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
            return top_candidates + results[len(top_candidates) :]
        except Exception as e:
            logger.error("Cross-encoder rerank failed: %s", e)
            return results

    def cap_per_domain(self, results: List[EvidenceResult]) -> List[EvidenceResult]:
        capped: List[EvidenceResult] = []
        domain_counts: Dict[str, int] = {}
        for r in results:
            domain = self._domain_from_url(r.get("url", ""))
            count = domain_counts.get(domain, 0)
            if count >= self.per_domain_cap:
                continue
            domain_counts[domain] = count + 1
            capped.append(r)
        return capped

    # --- Stance detection ---
    def attach_stance(self, results: List[EvidenceResult], claim_text: str) -> List[EvidenceResult]:
        for r in results:
            stance = "support"
            if self.use_nli_stance:
                stance = self._infer_stance_nli(claim_text, r.get("title", ""), r.get("snippet", "")) or "support"
            if stance == "support":
                stance = self._infer_stance_heuristic(claim_text, r.get("title", ""), r.get("snippet", ""), r.get("query", ""))
            r["stance"] = stance
        return results

    def _infer_stance_nli(self, claim_text: str, title: str, snippet: str) -> Optional[str]:
        if not claim_text:
            return None
        if not self.nli_model and not self.nli_loaded:
            try:
                self.nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
                self.nli_loaded = True
            except Exception as e:
                logger.warning("NLI model load failed: %s", e)
                self.nli_loaded = True
                self.nli_model = None
        if not self.nli_model:
            return None

        text = (title or "") + " " + (snippet or "")
        pairs = [(claim_text, text)]
        try:
            logits = self.nli_model.predict(pairs, apply_softmax=True)[0]
            # Model output order: entailment, neutral, contradiction (common for NLI CE)
            entail, neutral, contra = logits
            if contra > max(entail, neutral):
                return "contradict"
            if entail >= contra:
                return "support"
            return "support"
        except Exception as e:
            logger.debug("NLI stance failed: %s", e)
            return None

    def _infer_stance_heuristic(self, claim_text: str, title: str, snippet: str, query: str) -> str:
        text_blob = " ".join([query or "", title or "", snippet or ""]).lower()
        claim_terms = set(claim_text.lower().split())
        contradict_markers = [
            "debunk",
            "fake",
            "hoax",
            "false",
            "refute",
            "refuted",
            "not true",
            "no evidence",
            "no link",
            "no correlation",
            "does not cause",
            "do not cause",
            "not caused",
            "myth",
            "disproven",
            "disproved",
            "misinformation",
            "ruled out",
            "ruled-out",
        ]
        if any(m in text_blob for m in contradict_markers):
            return "contradict"

        neg_patterns = [r"\bnot\b", r"\bdoes\s+not\b", r"\bno\b", r"\bnever\b"]
        if any(re.search(p, text_blob) for p in neg_patterns):
            if any(term in text_blob for term in claim_terms):
                return "contradict"

        return "support"

    # --- Utilities ---
    def _domain_from_url(self, url: str) -> str:
        try:
            parsed = requests.utils.urlparse(url)
            domain = parsed.netloc or ""
            if domain.startswith("www."):
                domain = domain[4:]
            return domain.lower()
        except Exception:
            return ""

    def _get_from_cache(self, query: str) -> Optional[List[EvidenceResult]]:
        if not self.use_cache:
            return None
        return self.memory_cache.get(query)

    def _save_to_cache(self, query: str, results: List[EvidenceResult]):
        if not self.use_cache:
            return
        self.memory_cache[query] = results
