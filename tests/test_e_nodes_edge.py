import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta, timezone

from nodes.E_nodes.e1_web_evidence import WebSearcher, run as e1_run
from nodes.E_nodes import e2_source_reliability
from nodes.E_nodes import e3_claim_evidence_scorer
from nodes.utils import schema


class TestE1EdgeCases(unittest.TestCase):
    def test_rank_fallback_when_embedding_unavailable(self):
        searcher = WebSearcher(debug=True, use_cache=False)
        searcher.embedding_model = None
        searcher.embedding_loaded = True  # force fallback path
        results = [
            {"title": "Alpha", "snippet": "foo bar", "relevance_score": 0.0},
            {"title": "Beta", "snippet": "foo baz", "relevance_score": 0.0},
        ]
        ranked = searcher.rank_results(results, "foo qux")
        self.assertEqual(ranked[0]["ranking_mode"], "fallback")
        self.assertGreaterEqual(ranked[0]["relevance_score"], ranked[1]["relevance_score"])

    def test_near_duplicate_deduplication(self):
        searcher = WebSearcher(debug=True, use_cache=False)
        results = [
            {"url": "http://a.com", "title": "Title one", "snippet": "abc def"},
            {"url": "http://b.com", "title": "Title one", "snippet": "abc def"},  # near dup
            {"url": "http://c.com", "title": "Different", "snippet": "ghi"},
        ]
        deduped = searcher.deduplicate(results)
        self.assertEqual(len(deduped), 2)

    def test_stance_detection_from_query(self):
        state = {
            "claims": [{"claim_text": "Earth is flat", "id": "1"}],
            "use_cache": False,
        }
        with patch.object(WebSearcher, "construct_queries", return_value=["earth is flat debunked"]), \
             patch.object(WebSearcher, "search_robust", return_value=[{"url": "u", "title": "t", "snippet": "s", "source": "ddg", "relevance_score": 0.1}]), \
             patch.object(WebSearcher, "deduplicate", side_effect=lambda x: x), \
             patch.object(WebSearcher, "rank_results", side_effect=lambda x, q: x):
            result_state = e1_run(state)
        ev = result_state["evidence"][0]
        self.assertEqual(ev["stance"], "contradict")

    def test_mixed_claim_inputs_normalization(self):
        state = {
            "claims": [
                {"text": "Alpha claim", "confidence": 0.9, "source": "transcript"},
                "Beta claim string"
            ],
            "use_cache": False,
        }
        with patch.object(WebSearcher, "construct_queries", return_value=["alpha debunked"]), \
             patch.object(WebSearcher, "search_robust", return_value=[{"url": "u1", "title": "t1", "snippet": "s1", "source": "ddg", "relevance_score": 0.1}]), \
             patch.object(WebSearcher, "deduplicate", side_effect=lambda x: x), \
             patch.object(WebSearcher, "rank_results", side_effect=lambda x, q: x):
            result_state = e1_run(state)
        self.assertIn("claims", result_state)
        self.assertEqual(len(result_state["claims"]), 2)
        # evidence should carry claim_confidence/source when present
        ev = result_state["evidence"][0]
        self.assertIn("claim_confidence", ev)
        self.assertEqual(ev["claim_source"], "transcript")

    def test_cross_query_deduplication_same_url(self):
        state = {
            "claims": [{"claim_text": "Gamma claim"}],
            "use_cache": False,
        }
        def fake_search(query):
            return [{"url": "http://same.com", "title": query, "snippet": "body", "source": "ddg", "relevance_score": 0.1}]
        orig_dedup = WebSearcher.deduplicate
        with patch.object(WebSearcher, "construct_queries", return_value=["q1", "q2"]), \
             patch.object(WebSearcher, "search_robust", side_effect=fake_search), \
             patch.object(WebSearcher, "deduplicate", side_effect=lambda self, x: orig_dedup(self, x)), \
             patch.object(WebSearcher, "rank_results", side_effect=lambda x, q: x):
            result_state = e1_run(state)
        self.assertEqual(len(result_state["evidence"]), 1)

    def test_stance_from_snippet_negation(self):
        state = {
            "claims": [{"claim_text": "Sugar causes cancer", "id": "1"}],
            "use_cache": False,
        }
        results = [{
            "url": "u1", "title": "Study shows sugar does not cause cancer",
            "snippet": "This study shows sugar does not cause cancer", "source": "ddg", "relevance_score": 0.1
        }]
        with patch.object(WebSearcher, "construct_queries", return_value=["sugar causes cancer"]), \
             patch.object(WebSearcher, "search_robust", return_value=results), \
             patch.object(WebSearcher, "deduplicate", side_effect=lambda x: x), \
             patch.object(WebSearcher, "rank_results", side_effect=lambda x, q: x):
            result_state = e1_run(state)
        self.assertEqual(result_state["evidence"][0]["stance"], "contradict")


class TestE2EdgeCases(unittest.TestCase):
    def test_freshness_bonus(self):
        trusted = {"high_trust": [], "medium_trust": []}
        recent_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        item = {"url": "https://example.com", "claim_text": "c1", "date": recent_date}
        score = e2_source_reliability.calculate_reliability_score(
            item, trusted, {}, lightweight=False, enable_http_status_check=False
        )
        self.assertTrue(any("Fresh content" in d for d in score["details"]))

    def test_domain_age_bonus(self):
        trusted = {"high_trust": [], "medium_trust": []}
        item = {"url": "https://oldsite.com", "claim_text": "c1"}
        with patch.object(e2_source_reliability, "_estimate_domain_age_years", return_value=10):
            score = e2_source_reliability.calculate_reliability_score(item, trusted, {}, lightweight=False)
        self.assertTrue(any("Domain age" in d for d in score["details"]))

    def test_consensus_ratio_applied(self):
        trusted = {"high_trust": [], "medium_trust": []}
        evidence = [
            {"url": "https://a.com", "claim_text": "c1"},
            {"url": "https://b.com", "claim_text": "c1"},
            {"url": "https://c.com", "claim_text": "c1"},
            {"url": "https://d.com", "claim_text": "c1"},
        ]
        state = {"evidence": evidence, "lightweight_reliability": True}
        result = e2_source_reliability.run(state)
        self.assertTrue(any("Consensus" in d for d in result["evidence"][0]["reliability_details"]))

    def test_cap_at_one_with_multiple_bonuses(self):
        trusted = {"high_trust": ["trusted.com"], "medium_trust": []}
        item = {
            "url": "https://trusted.com/path",
            "claim_text": "c1",
            "date": (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d"),
        }
        with patch.object(e2_source_reliability, "_estimate_domain_age_years", return_value=10):
            score = e2_source_reliability.calculate_reliability_score(item, trusted, {"c1": 1.0}, lightweight=False)
        self.assertAlmostEqual(score["score"], 1.0)

    def test_lightweight_skips_expensive_checks(self):
        trusted = {"high_trust": [], "medium_trust": []}
        item = {"url": "https://example.com", "claim_text": "c1"}
        with patch.object(e2_source_reliability, "_estimate_domain_age_years", return_value=5) as mock_age:
            score = e2_source_reliability.calculate_reliability_score(item, trusted, {}, lightweight=True)
        self.assertFalse(any("Domain age" in d for d in score["details"]))
        mock_age.assert_not_called()

    def test_social_penalty(self):
        trusted = {"high_trust": [], "medium_trust": []}
        item = {"url": "https://www.youtube.com/watch?v=123", "claim_text": "c1"}
        score = e2_source_reliability.calculate_reliability_score(item, trusted, {}, lightweight=True)
        self.assertLess(score["score"], 0.5)
        self.assertTrue(any("UGC/Social" in d for d in score["details"]))


class TestE3EdgeCases(unittest.TestCase):
    def test_all_contradiction_reduces_score(self):
        claims = [{"claim_text": "Claim 1"}]
        evidence = [
            {"claim_text": "Claim 1", "reliability_score": 0.9, "stance": "contradict"},
            {"claim_text": "Claim 1", "reliability_score": 0.8, "stance": "contradict"},
        ]
        state = {"claims": claims, "evidence": evidence}
        result = e3_claim_evidence_scorer.run(state)
        scored = result["claims"][0]
        self.assertEqual(scored["evidence_score"], 0.0)
        self.assertEqual(scored["verdict"], "False")

    def test_mixed_support_and_contradiction(self):
        claims = [{"claim_text": "Claim 1"}]
        evidence = [
            {"claim_text": "Claim 1", "reliability_score": 0.9, "stance": "support", "url": "https://a.com"},
            {"claim_text": "Claim 1", "reliability_score": 0.4, "stance": "contradict", "url": "https://b.com"},
        ]
        state = {"claims": claims, "evidence": evidence}
        result = e3_claim_evidence_scorer.run(state)
        scored = result["claims"][0]
        self.assertGreater(scored["evidence_score"], 0)
        self.assertIn(scored["verdict"], ["Possible", "Likely", "Highly Likely"])

    def test_diversity_zero_without_urls(self):
        claims = [{"claim_text": "Claim 1"}]
        evidence = [
            {"claim_text": "Claim 1", "reliability_score": 0.9, "stance": "support"},
            {"claim_text": "Claim 1", "reliability_score": 0.8, "stance": "support"},
        ]
        state = {"claims": claims, "evidence": evidence}
        result = e3_claim_evidence_scorer.run(state)
        scored = result["claims"][0]
        self.assertLessEqual(scored["evidence_score"], 0.7)

    def test_claim_id_mapping_with_diverse_sources(self):
        claims = [{"id": "c1", "claim_text": "Claim 1"}]
        evidence = [
            {"claim_id": "c1", "reliability_score": 0.9, "stance": "support", "url": "https://a.com"},
            {"claim_id": "c1", "reliability_score": 0.6, "stance": "support", "url": "https://b.com"},
            {"claim_id": "c1", "reliability_score": 0.4, "stance": "contradict", "url": "https://c.com"},
        ]
        state = {"claims": claims, "evidence": evidence}
        result = e3_claim_evidence_scorer.run(state)
        scored = result["claims"][0]
        self.assertGreater(scored["evidence_score"], 0.0)
        self.assertIn(scored["verdict"], ["Possible", "Likely", "Highly Likely"])

    def test_missing_reliability_defaults_zero(self):
        claims = [{"claim_text": "Claim 1"}]
        evidence = [{"claim_text": "Claim 1"}]
        state = {"claims": claims, "evidence": evidence}
        result = e3_claim_evidence_scorer.run(state)
        scored = result["claims"][0]
        self.assertEqual(scored["evidence_score"], 0.0)
        self.assertEqual(scored["verdict"], "Unverified")


if __name__ == "__main__":
    unittest.main()
