# E-Nodes (Version 1 & 2 alignment)

Version 1 (ingestion → perception → claim extraction → evidence retrieval → source reliability → consistency/scoring) and Version 2 (A*/V*/C*/E*/LR) both land on the same E-node roles: pull web evidence, judge source quality, and aggregate into a claim-level verdict.

## What each E node does
- **E1 – Evidence retrieval (`e1_web_evidence.py`)**: Builds support/contradict queries, searches (Serper → Google CSE → DuckDuckGo), deduplicates near-duplicates, ranks by embeddings with cross-encoder rerank, caps per-domain hits, and assigns stance via NLI with heuristic fallback.
- **E2 – Source reliability (`e2_source_reliability.py`)**: Heuristic scorer using tldextract domain parsing, trusted-source tiers, HTTPS/authorship signals, optional domain-age/freshness, social/UGC penalties, consensus boost, and optional HTTP status check. Outputs `reliability_score` and `reliability_details`.
- **E3 – Claim evidence scorer (`e3_claim_evidence_scorer.py`)**: Aggregates reliability-weighted support vs. contradiction (length-aware), adds median reliability and domain diversity, and emits a verdict. Contradiction dominance can force a False verdict even with moderate scores.

## Inputs and outputs
- **Input to E1:** `state["claims"]` from C3 (list of dicts with at least `claim_text`; source/confidence carried through when present).
- **E1 output:** `state["evidence"]` as a flat list with claim metadata, stance (support/contradict), rank, query, timestamp.
- **E2 output:** Same list with `reliability_score` and `reliability_details`.
- **E3 output:** `state["claims"]` updated with `evidence_score`, `verdict`, and `evidence_count`; these feed the downstream consistency/scoring/logreg stage described in Version 1.

## Feature hooks (used downstream)
- **E3-derived features:** `claim_support_ratio`, `median_source_reliability`, `asr_ocr_consistency` and related evidence scores feed the Version 1 “consistency & scoring” / LR node.

## Data flow (Version 1)
1. C3 (claim extraction) → **E1** (evidence retrieval)
2. **E1** → **E2** (source reliability)
3. **E1 + E2 + C3** → **E3** (claim-level score/verdict)
4. E3 features go into the logistic regression/consistency block alongside other modal features.
