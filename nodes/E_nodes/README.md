# E-Nodes: Evidence Retrieval & Analysis

This module implements the Evidence (E) nodes of the Fake Video Detection Engine. These nodes are responsible for gathering external evidence, assessing source reliability, and scoring the veracity of claims extracted from the video.

## Components

### 1. E1: Web Evidence Retrieval (`e1_web_evidence.py`)
**Goal:** Retrieve relevant articles, fact-checks, and web pages to support or refute claims.

**Key Features:**
- **Smart Query Construction:** Generates multiple query variants for each claim to maximize search coverage:
  - *Entity-focused:* `"{Person}" {Action}`
  - *Context-focused:* `"{Action}" {Location} {Time}`
  - *Verification-oriented:* `"{Claim}" fact check`
- **Multi-API Fallback Strategy:**
  1. **Serper Dev API:** (Best quality, structured data)
  2. **Google Custom Search API:** (High quality, reliable)
  3. **DuckDuckGo:** (Free, privacy-focused, rate-limited)
- **Result Processing:**
  - **Deduplication:** Removes duplicate URLs and near-duplicate content snippets.
  - **Semantic Ranking:** Uses `sentence-transformers` to rank results by semantic similarity to the claim, ensuring the most relevant evidence is prioritized.
  - **Caching:** Caches search results (Redis or in-memory) to reduce API costs and latency.

### 2. E2: Source Reliability (`e2_source_reliability.py`)
**Goal:** Evaluate the credibility and trustworthiness of the sources found in E1.

**Key Features:**
- **Tiered Scoring System:** Utilizes `trusted_sources.json` to categorize domains into tiers:
  - **Tier 1 (1.0):** Top-tier agencies (e.g., Reuters, AP, BBC).
  - **Tier 2 (0.8):** Major reliable news (e.g., NYT, WaPo).
  - **Tier 3 (0.6):** Regional/Opinionated but factual.
  - **Tier 4 (0.4):** Tabloids/Sensationalist.
  - **Tier 5 (0.2):** State propaganda/Unreliable.
  - **Tier 6 (0.0):** Known fake news/Satire.
- **Heuristic Signals:** Adjusts the base score using real-time checks:
  - **SSL/HTTPS:** (+0.05) Basic security check.
  - **"About" Page:** (+0.05) Transparency signal.
  - **Domain Age:** (Up to +0.08) Older domains are generally more trusted.
  - **Author Attribution:** (+0.05) Presence of bylines.
- **Consensus Boost:** Increases the score if a majority of retrieved sources for a claim are deemed reliable.

### 3. E3: Claim Evidence Scorer (`e3_claim_evidence_scorer.py`)
**Goal:** Compute a comprehensive set of features to determine the likelihood of the claims being true based on the gathered evidence.

**Key Features:**
- **Semantic Analysis:** Uses `sentence-transformers` to measure:
  - **Semantic Coherence:** Do the evidence snippets tell a consistent story?
  - **ASR-OCR Consistency:** Does the audio transcript match the on-screen text?
- **Entity Matching (NER):** Uses `spaCy` to extract and match entities (PERSON, ORG, GPE, DATE) between the claim and the evidence.
- **Contradiction Detection:** Detects negation and refutation patterns in the evidence snippets.
- **Calculated Features:**
  - `claim_support_ratio`: Fraction of claims supported by reliable sources.
  - `median_source_reliability`: The central tendency of source quality.
  - `confidence_score`: A meta-score indicating the system's confidence in its assessment.

## Implementation Details

### E1: Web Evidence Retrieval
- **Query Generation:** The `EnhancedWebSearcher` class uses a template-based approach to generate 4 distinct query types per claim. This ensures that if a specific phrasing fails, broader or more targeted queries might succeed.
- **Deduplication Logic:** Results are deduplicated based on URL (exact match) and Content Similarity (Jaccard index of title + snippet tokens > 0.7).
- **Ranking:** Results are re-ranked using a weighted score:
  - **Semantic Similarity (0.5):** Cosine similarity between the claim embedding and result snippet embedding (using `all-MiniLM-L6-v2`).
  - **Search Position (0.3):** Decaying score based on original rank from the search engine.
  - **Recency (0.2):** Decaying score based on publication date (newer is better).

### E2: Source Reliability
- **Tiered Database:** The core logic relies on `trusted_sources.json`, which maps domains to a 6-tier system.
- **Heuristic Adjustments:**
  - **SSL:** Checks `urllib.parse` scheme.
  - **About Page:** Performs a HEAD request to `/about`, `/about-us`, etc.
  - **Domain Age:** Uses `python-whois` to fetch creation date. Domains < 1 year old get a penalty; > 5 years get a boost.
  - **Consensus:** If > 50% of retrieved sources for a claim have a score > 0.7, a `0.1` boost is applied to the final reliability score.

### E3: Claim Scorer
- **Semantic Consistency:**
  - Uses `sentence-transformers` to encode all evidence snippets.
  - Calculates the average pairwise cosine similarity between all snippets. High similarity (> 0.5) implies a consistent narrative.
- **Entity Matching:**
  - Uses `spaCy` (`en_core_web_sm`) to extract named entities.
  - Calculates the Jaccard overlap of unique entities between the Claim and the Evidence.
- **Confidence Score:**
  - A heuristic meta-score derived from: `(0.3 * Source Count) + (0.3 * Median Reliability) + (0.2 * Consistency) + (0.2 * Non-Contradiction)`.


## Scoring Formulas

### E1: Relevance Ranking
The final `relevance_score` for each search result is a weighted sum:
`Score = (0.5 * Semantic) + (0.3 * Position) + (0.2 * Recency)`

- **Semantic**: Cosine similarity between Claim embedding and Result (Title + Snippet) embedding.
- **Position**: Decay based on search engine rank: `max(0, 1 - (rank / 50))`.
- **Recency**: Decay based on article age (in days): `max(0, 1 - (age / 365))`.

### E2: Reliability Score
`Reliability = Base + Sum(Heuristics) + Boost`

- **Base Score**: From `trusted_sources.json` (e.g., Tier 1 = 1.0, Tier 5 = 0.2).
- **Heuristics**:
  - **SSL**: +0.05
  - **About Page**: +0.05
  - **Author**: +0.05
  - **Domain Age**: Up to +0.08 (scaled linearly over 5 years).
  - **Freshness**: Up to +0.07 (for content < 7 days old).
- **Consensus Boost**: +0.05 if >60% of sources for the claim are reliable (>0.7).

### E3: Confidence Score
A meta-metric representing the system's certainty:
`Confidence = Vol + Qual + Cons + Contra`

- **Volume**: `min(count / 10, 0.3)` (More sources = better).
- **Quality**: `MedianReliability * 0.3`.
- **Consistency**: `EvidenceConsistency * 0.2` (Low variance in source scores).
- **Non-Contradiction**: `(1 - ContradictionScore) * 0.2`.


## Data Flow & Architecture
The implementation strictly follows the directed acyclic graph (DAG) defined in `main.py`:

1.  **C3 (Claim Extraction) -> E1**:
    -   **Input**: `state["claims"]` (List of dictionaries: who, what, when, where).
    -   **Output**: `state["evidence"]` (List of evidence objects populated with search results).

2.  **E1 -> E2**:
    -   **Input**: `state["evidence"]` (Raw search results from Serper/Google/DDG).
    -   **Output**: `state["evidence"]` (Enriched with `reliability_score` and `reliability_factors` for each result).

3.  **E2 -> E3**:
    -   **Input**: `state["evidence"]` (Reliability-scored evidence).
    -   **Output**: `state["features"]` (Aggregated numerical scores like `claim_support_ratio`, `confidence_score`).

4.  **V-Nodes (V2, A2) -> E3**:
    -   **Input**: `state["ocr_results"]` (from V2) and `state["transcript"]` (from A2).
    -   **Output**: Used to calculate `asr_ocr_consistency`.

## Sample Test Output
The following output demonstrates the production-ready E-nodes in action:

```text
Testing E Nodes (Production-Ready Suite)...

--- Running E1 ---
Node E1: Retrieving Web Evidence (Production-Ready)...
⚠️  Redis cache unavailable, using in-memory

============================================================
Processing Claim 1/2
============================================================

📝 Query: 'Person A did something London yesterday'
   🔍 Trying DuckDuckGo...
   ✅ DuckDuckGo: 2 results

📝 Query: '"Person A" did something yesterday'
   🔍 Trying DuckDuckGo...
   ✅ DuckDuckGo: 2 results

📝 Query: 'did something yesterday London'
   🔍 Trying DuckDuckGo...
   ✅ DuckDuckGo: 1 results

📝 Query: 'Person A did something fact check'
   🔍 Trying DuckDuckGo...
   ✅ DuckDuckGo: 2 results

✅ Retrieved 3 unique results
   1. Person A did something confirmed...
   2. Lies about Person A...
   3. Person B statement...

============================================================
Processing Claim 2/2
============================================================

📝 Query: 'Person B said something today'
   🔍 Trying DuckDuckGo...
   ✅ DuckDuckGo: 1 results

📝 Query: '"Person B" said something today'
   🔍 Trying DuckDuckGo...
   ✅ DuckDuckGo: 1 results

📝 Query: 'said something today'
   🔍 Trying DuckDuckGo...
   ✅ DuckDuckGo: 1 results

📝 Query: 'Person B said something fact check'
   🔍 Trying DuckDuckGo...
   ✅ DuckDuckGo: 1 results

✅ Retrieved 1 unique results
   1. Person B statement...

✅ E1: Retrieved evidence for 2 claims.
E1 Passed (Mocked DDG returned results)

--- Running E2 ---
Node E2: Assessing Source Reliability (Production-Ready)...
✅ Loaded 135 trusted domains

   📊 bbc.com: 1.000
      • database_match: +1.000
      • ssl_valid: +0.050

   📊 fake-news-network.xyz: 0.250
      • database_match: +0.200
      • ssl_valid: +0.050

   📊 reuters.com: 1.000
      • database_match: +1.000
      • ssl_valid: +0.050

   📊 reuters.com: 1.000
      • database_match: +1.000
      • ssl_valid: +0.050
   🔗 Applied consensus boost: 2/3 reliable sources
✅ E2: Reliability assessment complete.
BBC Score: 1.0
Fake News Score: 0.3
E2 Passed (Reliability Scores Correct)

--- Running E3 ---
Node E3: Scoring Claim Evidence (Production-Ready)...

============================================================
📊 FINAL FEATURE SCORES
============================================================
   claim_support_ratio: 1.000
   median_source_reliability: 1.000
   evidence_consistency: 0.835
   entity_match_score: 0.545
   temporal_consistency: 0.500
   spatial_consistency: 0.333
   asr_ocr_consistency: 0.300
   semantic_coherence: 0.500
   source_diversity: 0.750
   top_source_avg_score: 0.825
   contradiction_score: 0.250
   confidence_score: 0.917
============================================================

✅ E3: Scoring complete.
Features: {
  "claim_support_ratio": 1.0,
  "median_source_reliability": 1.0,
  "evidence_consistency": 0.835,
  "entity_match_score": 0.545,
  "temporal_consistency": 0.5,
  "spatial_consistency": 0.333,
  "asr_ocr_consistency": 0.3,
  "semantic_coherence": 0.5,
  "source_diversity": 0.75,
  "top_source_avg_score": 0.825,
  "contradiction_score": 0.25,
  "confidence_score": 0.917
}
E3 Passed (Features Calculated Correctly)

All Production-Ready E-Node tests passed!
```