import unittest
from unittest.mock import patch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nodes.E_nodes import e2_source_reliability


class TestE2Reliability(unittest.TestCase):
    def test_get_domain_registered_only(self):
        self.assertEqual(e2_source_reliability.get_domain("https://www.example.com/page"), "example.com")
        # Registered domain should drop subdomain
        self.assertEqual(e2_source_reliability.get_domain("http://sub.domain.org"), "domain.org")
        self.assertEqual(e2_source_reliability.get_domain("invalid-url"), "")

    def test_heuristic_scores(self):
        trusted = {"high_trust": ["trusted.com"], "medium_trust": []}
        # Base 0.5 + high trust 0.3 + HTTPS 0.05 = 0.85
        item = {"url": "https://trusted.com/news", "claim_text": "c1"}
        score = e2_source_reliability.calculate_reliability_score(item, trusted, {}, lightweight=True)
        self.assertAlmostEqual(score["score"], 0.85)

    def test_social_penalty(self):
        trusted = {"high_trust": [], "medium_trust": []}
        item = {"url": "https://www.youtube.com/watch?v=123", "claim_text": "c1"}
        score = e2_source_reliability.calculate_reliability_score(item, trusted, {}, lightweight=True)
        self.assertLess(score["score"], 0.5)
        self.assertTrue(any("UGC" in d or "Social" in d for d in score["details"]))

    def test_consensus_boost_applied(self):
        trusted = {"high_trust": [], "medium_trust": []}
        evidence = [
            {"url": "https://a.com", "claim_text": "c1"},
            {"url": "https://b.com", "claim_text": "c1"},
            {"url": "https://c.com", "claim_text": "c1"},
        ]
        state = {"evidence": evidence}
        result_state = e2_source_reliability.run(state)
        scored = result_state["evidence"][0]
        self.assertTrue(any("Consensus" in d for d in scored["reliability_details"]))


if __name__ == "__main__":
    unittest.main()
