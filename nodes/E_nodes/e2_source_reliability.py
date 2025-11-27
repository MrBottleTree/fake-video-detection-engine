import json
import os
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import requests
import tldextract

from nodes.utils.schema import flatten_evidence

SOCIAL_DOMAINS = [
    "youtube.com",
    "youtu.be",
    "instagram.com",
    "facebook.com",
    "fb.com",
    "tiktok.com",
    "x.com",
    "twitter.com",
    "reddit.com",
    "medium.com",
    "bit.ly",
    "tinyurl.com",
]

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_domain_age_cache: Dict[str, float] = {}


def load_trusted_sources(assets_dir: str = "assets") -> Dict[str, List[str]]:
    """Loads trusted sources from a JSON file."""
    try:
        possible_paths = [
            os.path.join("assets", "trusted_sources.json"),
            os.path.join(os.getcwd(), "assets", "trusted_sources.json"),
            os.path.join(os.path.dirname(__file__), "..", "..", "assets", "trusted_sources.json"),
        ]
        file_path = next((p for p in possible_paths if os.path.exists(p)), None)
        if not file_path:
            logger.warning("trusted_sources.json not found. Using empty lists.")
            return {"high_trust": [], "medium_trust": []}
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading trusted sources: {e}")
        return {"high_trust": [], "medium_trust": []}


def get_domain(url: str) -> str:
    """Extract the registered domain using tldextract to avoid subdomain spoofing."""
    try:
        extracted = tldextract.extract(url)
        if not extracted.domain:
            return ""
        domain = ".".join(part for part in [extracted.domain, extracted.suffix] if part)
        return domain.lower()
    except Exception:
        return ""


def _check_http_status(url: str, timeout: float = 2.0) -> Optional[int]:
    try:
        resp = requests.head(url, allow_redirects=True, timeout=timeout)
        return resp.status_code
    except Exception:
        return None


def _estimate_domain_age_years(domain: str) -> float:
    """Approximate domain age in years with caching; returns 0 if unknown/unavailable."""
    if not domain:
        return 0.0
    if domain in _domain_age_cache:
        return _domain_age_cache[domain]
    try:
        import whois  # type: ignore
    except Exception:
        _domain_age_cache[domain] = 0.0
        return 0.0
    try:
        data = whois.whois(domain)
        created = data.creation_date
        if isinstance(created, list):
            created = created[0]
        if not created:
            _domain_age_cache[domain] = 0.0
            return 0.0
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        now = datetime.now(timezone.utc)
        if not created.tzinfo:
            created = created.replace(tzinfo=timezone.utc)
        age_years = max(0.0, (now - created).days / 365.25)
        _domain_age_cache[domain] = age_years
        return age_years
    except Exception:
        _domain_age_cache[domain] = 0.0
        return 0.0


def _parse_date(date_str: str):
    if not date_str:
        return None
    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%d-%m-%Y",
        "%Y-%m-%dT%H:%M:%S",
        "%a, %d %b %Y %H:%M:%S %Z",
    ):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def _has_author_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in ["by ", "reported by", "author", "staff writer"])


def calculate_reliability_score(
    evidence_item: Dict[str, Any],
    trusted_sources: Dict[str, List[str]],
    claim_consensus_map: Dict[str, float],
    *,
    lightweight: bool = True,
    enable_http_status_check: bool = False,
) -> Dict[str, Any]:
    url = evidence_item.get("url", "")
    claim_text = evidence_item.get("claim_text", "")

    score = 0.5
    details = []

    if not url:
        return {"score": 0.0, "details": ["No URL provided"]}

    domain = get_domain(url)
    scheme = requests.utils.urlparse(url).scheme

    if domain.endswith(".gov") or domain.endswith(".mil"):
        score += 0.4
        details.append("Government/Military domain (+0.4)")
    elif domain.endswith(".edu"):
        score += 0.3
        details.append("Educational domain (+0.3)")

    if any(domain == trusted or domain.endswith("." + trusted) for trusted in trusted_sources.get("high_trust", [])):
        score += 0.3
        details.append("High trust source (+0.3)")
    elif any(domain == trusted or domain.endswith("." + trusted) for trusted in trusted_sources.get("medium_trust", [])):
        score += 0.1
        details.append("Medium trust source (+0.1)")

    if scheme == "https":
        score += 0.05
        details.append("Secure protocol (HTTPS) (+0.05)")
    else:
        score -= 0.05
        details.append("Insecure protocol (HTTP) (-0.05)")

    text_blob = f"{evidence_item.get('title','')} {evidence_item.get('snippet','')}"
    if _has_author_marker(text_blob):
        score += 0.05
        details.append("Author/byline signal (+0.05)")

    if not lightweight and domain:
        age_years = _estimate_domain_age_years(domain)
        if age_years:
            age_bonus = min(0.08, (age_years / 5.0) * 0.08)
            score += age_bonus
            details.append(f"Domain age ~{age_years:.1f}y (+{age_bonus:.3f})")

    date_val = evidence_item.get("date")
    parsed_date = _parse_date(date_val) if isinstance(date_val, str) else None
    if not lightweight and parsed_date:
        days_old = max(0, (datetime.now(timezone.utc) - parsed_date).days)
        freshness = max(0.0, 1 - (days_old / 7))
        freshness_bonus = min(0.07, freshness * 0.07)
        if freshness_bonus > 0:
            score += freshness_bonus
            details.append(f"Fresh content ({days_old} days old) (+{freshness_bonus:.3f})")

    claim_key = evidence_item.get("claim_id") or claim_text
    consensus_ratio = claim_consensus_map.get(claim_key, 0.0)
    if consensus_ratio >= 0.6:
        score += 0.05
        details.append(f"Consensus boost ({consensus_ratio:.2f}) (+0.05)")
    if consensus_ratio:
        evidence_item["consensus_ratio"] = consensus_ratio

    if any(domain == s or domain.endswith("." + s) or domain.endswith(s) for s in SOCIAL_DOMAINS):
        score -= 0.15
        details.append("UGC/Social source (-0.15)")

    if enable_http_status_check:
        status = _check_http_status(url)
        if status and not (200 <= status < 400):
            score -= 0.05
            details.append(f"Non-OK HTTP status {status} (-0.05)")

    final_score = min(1.0, max(0.0, score))
    return {"score": final_score, "details": details}


def run(state: dict) -> dict:
    """
    E2 Node: Source Reliability Scorer

    Input:
      - Preferred: flat evidence list (E1 output) with claim metadata
      - Legacy per-claim objects with nested "results" also supported

    Config:
      - state["lightweight_reliability"]: skip expensive checks (whois/about/freshness) when True (default)
      - state["enable_http_status_check"]: optional HEAD status check (default False)
    """
    print("--- E2: Source Reliability ---")
    evidence_list = state.get("evidence", [])

    if not evidence_list:
        print("No evidence found to score.")
        return state

    evidence_list = flatten_evidence(evidence_list)
    lightweight = state.get("lightweight_reliability", True)
    enable_http_status_check = state.get("enable_http_status_check", False)

    trusted_sources = load_trusted_sources()

    claim_domains: Dict[str, set] = {}
    for item in evidence_list:
        claim_key = item.get("claim_id") or item.get("claim_text") or "unknown"
        url = item.get("url", "")
        domain = get_domain(url)
        if claim_key and domain:
            claim_domains.setdefault(claim_key, set()).add(domain)

    claim_consensus_map = {claim_key: min(1.0, len(domains) / 5) for claim_key, domains in claim_domains.items()}

    scored_evidence = []
    for item in evidence_list:
        new_item = item.copy()
        result = calculate_reliability_score(
            new_item,
            trusted_sources,
            claim_consensus_map,
            lightweight=lightweight,
            enable_http_status_check=enable_http_status_check,
        )

        new_item["reliability_score"] = result["score"]
        new_item["reliability_details"] = result["details"]

        scored_evidence.append(new_item)
        print(f"Scored {new_item.get('url', 'N/A')}: {result['score']:.2f}")

    state["evidence"] = scored_evidence
    return state
