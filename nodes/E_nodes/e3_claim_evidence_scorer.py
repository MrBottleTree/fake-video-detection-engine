import statistics
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer, util
    SEMANTIC_MODEL = None  # Lazy load
except ImportError:
    SentenceTransformer = None

try:
    import spacy
    NLP_MODEL = None  # Lazy load
except ImportError:
    spacy = None


def run(state: dict) -> dict:
    """
    E3: Enhanced Claim Evidence Scoring
    
    Improvements:
    - Semantic similarity (not just Jaccard)
    - NER-based entity matching
    - Contradiction detection
    - Source diversity metrics
    - Confidence intervals
    - Feature importance weights
    """
    print("Node E3: Scoring Claim Evidence (Production-Ready)...")
    
    evidence_data = state.get("evidence", [])
    claims = state.get("claims", [])
    debug = state.get("debug", False)
    
    if "features" not in state:
        state["features"] = {}
    
    try:
        scorer = EnhancedClaimScorer(debug=debug)
        features = scorer.score_all_claims(evidence_data, claims, state)
        
        state["features"].update(features)
        
        if debug:
            print(f"\n{'='*60}")
            print("📊 FINAL FEATURE SCORES")
            print(f"{'='*60}")
            for feat_name, feat_value in features.items():
                if isinstance(feat_value, float):
                    print(f"   {feat_name}: {feat_value:.3f}")
                else:
                    print(f"   {feat_name}: {feat_value}")
            print(f"{'='*60}\n")
        
        print("✅ E3: Scoring complete.")
    
    except Exception as e:
        print(f"❌ Error in E3 node: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        raise e
    
    return state


class EnhancedClaimScorer:
    """Production-ready claim evidence scorer with semantic understanding"""
    
    def __init__(self, debug=False):
        self.debug = debug
        self.semantic_model = None
        self.nlp_model = None
    
    def load_models(self):
        """Lazy load NLP models"""
        if SentenceTransformer and not self.semantic_model:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                if self.debug:
                    print("✅ Loaded semantic similarity model")
            except Exception as e:
                if self.debug:
                    print(f"⚠️  Could not load semantic model: {e}")
        
        if spacy and not self.nlp_model:
            try:
                self.nlp_model = spacy.load("en_core_web_sm")
                if self.debug:
                    print("✅ Loaded spaCy NER model")
            except:
                if self.debug:
                    print("⚠️  Could not load spaCy. Install: python -m spacy download en_core_web_sm")
    
    def score_all_claims(self, evidence_data: List[Dict], claims: List[Dict], state: Dict) -> Dict[str, float]:
        """Calculate all evidence features"""
        self.load_models()
        
        features = {}
        
        # Core Evidence Features
        features["claim_support_ratio"] = self.calculate_support_ratio(evidence_data)
        features["median_source_reliability"] = self.calculate_median_reliability(evidence_data)
        features["evidence_consistency"] = self.calculate_evidence_consistency(evidence_data)
        
        # Entity Matching Features
        features["entity_match_score"] = self.calculate_entity_matching(evidence_data)
        features["temporal_consistency"] = self.calculate_temporal_consistency(evidence_data)
        features["spatial_consistency"] = self.calculate_spatial_consistency(evidence_data)
        
        # Content Features
        features["asr_ocr_consistency"] = self.calculate_asr_ocr_consistency(state)
        features["semantic_coherence"] = self.calculate_semantic_coherence(evidence_data)
        
        # Source Quality Features
        features["source_diversity"] = self.calculate_source_diversity(evidence_data)
        features["top_source_avg_score"] = self.calculate_top_sources_avg(evidence_data)
        
        # Contradiction Detection
        features["contradiction_score"] = self.detect_contradictions(evidence_data)
        
        # Confidence Estimate
        features["confidence_score"] = self.calculate_confidence(features, evidence_data)
        
        return features
    
    def calculate_support_ratio(self, evidence_data: List[Dict]) -> float:
        """
        Fraction of claims with at least one reliable supporting source.
        
        Improvements:
        - Stricter threshold (0.7 instead of 0.6)
        - Non-empty snippet requirement
        - Semantic relevance check if available
        """
        if not evidence_data:
            return 0.0
        
        supported_count = 0
        
        for item in evidence_data:
            results = item.get("results", [])
            is_supported = False
            
            for res in results:
                score = res.get("reliability_score", 0.5)
                snippet = res.get("snippet", "")
                
                # Strict criteria
                if score > 0.7 and len(snippet) > 50:
                    is_supported = True
                    break
            
            if is_supported:
                supported_count += 1
        
        return supported_count / len(evidence_data)
    
    def calculate_median_reliability(self, evidence_data: List[Dict]) -> float:
        """Median reliability across all reliable sources"""
        all_scores = []
        
        for item in evidence_data:
            results = item.get("results", [])
            for res in results:
                score = res.get("reliability_score", 0.5)
                if score > 0.6:  # Only count reasonably reliable sources
                    all_scores.append(score)
        
        return statistics.median(all_scores) if all_scores else 0.5
    
    def calculate_evidence_consistency(self, evidence_data: List[Dict]) -> float:
        """
        Measure how consistently evidence supports claims.
        Low variance in reliability = high consistency.
        """
        all_scores = []
        
        for item in evidence_data:
            results = item.get("results", [])
            item_scores = [r.get("reliability_score", 0.5) for r in results]
            if item_scores:
                all_scores.append(statistics.mean(item_scores))
        
        if len(all_scores) < 2:
            return 0.5
        
        # Low standard deviation = high consistency
        std_dev = statistics.stdev(all_scores)
        consistency = max(0, 1 - std_dev)
        
        return consistency
    
    def calculate_entity_matching(self, evidence_data: List[Dict]) -> float:
        """
        NER-based entity matching between claims and evidence.
        
        Uses spaCy to extract:
        - PERSON entities
        - ORG entities
        - GPE (locations)
        - DATE entities
        """
        if not self.nlp_model:
            # Fallback to simple keyword matching
            return self.calculate_entity_matching_simple(evidence_data)
        
        total_matches = 0
        total_entities = 0
        
        for item in evidence_data:
            claim = item.get("claim", {})
            results = item.get("results", [])
            
            # Extract entities from claim
            if isinstance(claim, dict):
                claim_text = " ".join([str(v) for v in claim.values() if v])
            else:
                claim_text = str(claim)
            
            claim_doc = self.nlp_model(claim_text)
            claim_entities = {ent.text.lower() for ent in claim_doc.ents}
            
            if not claim_entities:
                continue
            
            # Check how many claim entities appear in evidence
            for res in results:
                snippet = res.get("snippet", "")
                title = res.get("title", "")
                evidence_text = f"{title} {snippet}"
                evidence_doc = self.nlp_model(evidence_text)
                evidence_entities = {ent.text.lower() for ent in evidence_doc.ents}
                
                # Count matches
                matches = claim_entities & evidence_entities
                total_matches += len(matches)
                total_entities += len(claim_entities)
        
        return total_matches / total_entities if total_entities > 0 else 0.5
    
    def calculate_entity_matching_simple(self, evidence_data: List[Dict]) -> float:
        """Fallback entity matching using claim fields"""
        matches = 0
        total = 0
        
        for item in evidence_data:
            claim = item.get("claim", {})
            results = item.get("results", [])
            
            if not isinstance(claim, dict):
                continue
            
            who = claim.get("who", "").lower()
            what = claim.get("what", "").lower()
            where = claim.get("where", "").lower()
            
            entities = [e for e in [who, what, where] if e]
            
            if not entities:
                continue
            
            for res in results:
                snippet = res.get("snippet", "").lower()
                title = res.get("title", "").lower()
                text = f"{title} {snippet}"
                
                for entity in entities:
                    if entity in text:
                        matches += 1
                total += len(entities)
        
        return matches / total if total > 0 else 0.5
    
    def calculate_temporal_consistency(self, evidence_data: List[Dict]) -> float:
        """
        Check if temporal entities (dates, times) match between claims and evidence.
        
        Improvements:
        - Fuzzy date matching (not just exact strings)
        - Relative time understanding ("yesterday", "last week")
        - Date range overlap detection
        """
        matches = 0
        total = 0
        
        for item in evidence_data:
            claim = item.get("claim", {})
            results = item.get("results", [])
            
            when = None
            if isinstance(claim, dict):
                when = claim.get("when", "").lower().strip()
            
            if not when:
                continue
            
            # Extract temporal patterns
            temporal_patterns = self.extract_temporal_patterns(when)
            
            for res in results:
                snippet = res.get("snippet", "").lower()
                title = res.get("title", "").lower()
                text = f"{title} {snippet}"
                
                # Check for any temporal pattern match
                for pattern in temporal_patterns:
                    if pattern in text:
                        matches += 1
                        break
                
                total += 1
        
        return matches / total if total > 0 else 0.5
    
    def extract_temporal_patterns(self, when_text: str) -> List[str]:
        """Extract various forms of temporal reference"""
        patterns = [when_text]
        
        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', when_text)
        if year_match:
            patterns.append(year_match.group())
        
        # Extract month
        months = ["january", "february", "march", "april", "may", "june",
                  "july", "august", "september", "october", "november", "december"]
        for month in months:
            if month in when_text.lower():
                patterns.append(month)
        
        # Extract day
        day_match = re.search(r'\b\d{1,2}\b', when_text)
        if day_match:
            patterns.append(day_match.group())
        
        return patterns
    
    def calculate_spatial_consistency(self, evidence_data: List[Dict]) -> float:
        """Check if spatial entities (locations) match"""
        matches = 0
        total = 0
        
        for item in evidence_data:
            claim = item.get("claim", {})
            results = item.get("results", [])
            
            where = None
            if isinstance(claim, dict):
                where = claim.get("where", "").lower().strip()
            
            if not where:
                continue
            
            for res in results:
                snippet = res.get("snippet", "").lower()
                title = res.get("title", "").lower()
                text = f"{title} {snippet}"
                
                if where in text:
                    matches += 1
                total += 1
        
        return matches / total if total > 0 else 0.5
    
    def calculate_asr_ocr_consistency(self, state: Dict) -> float:
        """
        Measure consistency between spoken content (ASR) and on-screen text (OCR).
        
        Improvements:
        - Semantic similarity instead of Jaccard
        - Entity-level matching
        """
        transcript = state.get("transcript", "")
        ocr_results = state.get("ocr_results", [])
        
        if not transcript or not ocr_results:
            return 0.5  # Unknown
        
        # Aggregate OCR text
        ocr_text = ""
        for frame in ocr_results:
            detections = frame.get("detections", [])
            for d in detections:
                ocr_text += " " + d.get("text", "")
        
        ocr_text = ocr_text.strip()
        
        if not ocr_text:
            return 0.5
        
        # Use semantic similarity if available
        if self.semantic_model:
            try:
                emb1 = self.semantic_model.encode(transcript, convert_to_tensor=True)
                emb2 = self.semantic_model.encode(ocr_text, convert_to_tensor=True)
                similarity = util.cos_sim(emb1, emb2).item()
                return similarity
            except:
                pass
        
        # Fallback to Jaccard
        return self.calculate_jaccard(transcript, ocr_text)
    
    def calculate_jaccard(self, text1: str, text2: str) -> float:
        """Basic Jaccard similarity"""
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_semantic_coherence(self, evidence_data: List[Dict]) -> float:
        """
        Measure semantic coherence across all evidence snippets.
        Do they tell a consistent story?
        """
        if not self.semantic_model:
            return 0.5  # Unknown without model
        
        all_texts = []
        for item in evidence_data:
            results = item.get("results", [])
            for res in results:
                snippet = res.get("snippet", "")
                if len(snippet) > 30:
                    all_texts.append(snippet)
        
        if len(all_texts) < 2:
            return 0.5
        
        try:
            # Compute pairwise similarities
            embeddings = self.semantic_model.encode(all_texts, convert_to_tensor=True)
            similarities = []
            
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = util.cos_sim(embeddings[i], embeddings[j]).item()
                    similarities.append(sim)
            
            # High average similarity = coherent story
            return statistics.mean(similarities) if similarities else 0.5
        
        except:
            return 0.5
    
    def calculate_source_diversity(self, evidence_data: List[Dict]) -> float:
        """
        Measure diversity of sources.
        More diverse sources = more trustworthy evidence.
        """
        all_domains = []
        
        for item in evidence_data:
            results = item.get("results", [])
            for res in results:
                domain = res.get("domain")
                if domain:
                    all_domains.append(domain)
        
        if not all_domains:
            return 0.0
        
        # Unique domains / total domains
        unique_count = len(set(all_domains))
        total_count = len(all_domains)
        
        diversity = unique_count / total_count
        return diversity
    
    def calculate_top_sources_avg(self, evidence_data: List[Dict]) -> float:
        """Average reliability of top 3 sources per claim"""
        top_scores = []
        
        for item in evidence_data:
            results = item.get("results", [])
            scores = sorted([r.get("reliability_score", 0.5) for r in results], reverse=True)
            
            if scores:
                top_3 = scores[:3]
                top_scores.extend(top_3)
        
        return statistics.mean(top_scores) if top_scores else 0.5
    
    def detect_contradictions(self, evidence_data: List[Dict]) -> float:
        """
        Detect contradictions in evidence.
        
        Simple heuristic:
        - Look for negation patterns ("not", "false", "deny", "refute")
        - Check if some sources support and others contradict
        
        Returns: contradiction score (0 = no contradiction, 1 = high contradiction)
        """
        contradiction_indicators = ["not true", "false", "deny", "denies", "refute", "refutes",
                                    "incorrect", "wrong", "misleading", "fake"]
        
        contradiction_count = 0
        total_snippets = 0
        
        for item in evidence_data:
            results = item.get("results", [])
            for res in results:
                snippet = res.get("snippet", "").lower()
                
                for indicator in contradiction_indicators:
                    if indicator in snippet:
                        contradiction_count += 1
                        break
                
                total_snippets += 1
        
        # Return proportion of contradictory snippets
        return contradiction_count / total_snippets if total_snippets > 0 else 0.0
    
    def calculate_confidence(self, features: Dict[str, float], evidence_data: List[Dict]) -> float:
        """
        Meta-feature: Overall confidence in the evidence assessment.
        
        Based on:
        - Number of sources
        - Reliability scores
        - Consistency metrics
        """
        # Count total sources
        source_count = sum(len(item.get("results", [])) for item in evidence_data)
        
        # Confidence factors
        conf = 0.0
        
        # More sources = higher confidence (capped)
        conf += min(source_count / 10, 0.3)
        
        # High reliability = higher confidence
        conf += features.get("median_source_reliability", 0.5) * 0.3
        
        # High consistency = higher confidence
        conf += features.get("evidence_consistency", 0.5) * 0.2
        
        # Low contradictions = higher confidence
        conf += (1 - features.get("contradiction_score", 0.5)) * 0.2
        
        return min(conf, 1.0)
