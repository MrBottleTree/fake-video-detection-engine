import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nodes.E_nodes import e2_source_reliability

class TestE2Reliability(unittest.TestCase):

    def setUp(self):
        self.mock_trusted_sources = {
            "high_trust": ["trusted.com"],
            "medium_trust": ["semi-trusted.com"]
        }

    def test_get_domain(self):
        self.assertEqual(e2_source_reliability.get_domain("https://www.example.com/page"), "example.com")
        self.assertEqual(e2_source_reliability.get_domain("http://sub.domain.org"), "sub.domain.org")
        self.assertEqual(e2_source_reliability.get_domain("invalid-url"), "")

    @patch('nodes.E_nodes.e2_source_reliability.load_trusted_sources')
    @patch('urllib.request.urlopen')
    def test_calculate_reliability_score(self, mock_urlopen, mock_load):
        mock_load.return_value = self.mock_trusted_sources
        
        # Mock "About" page check to fail by default (timeout or 404) to isolate other scores
        mock_urlopen.side_effect = Exception("Network error")

        # 1. Base Score (0.5) + HTTPS (0.1) = 0.6
        item1 = {"url": "https://unknown.com", "claim_text": "claim1"}
        score1 = e2_source_reliability.calculate_reliability_score(item1, self.mock_trusted_sources, {})
        self.assertAlmostEqual(score1["score"], 0.6)

        # 2. Gov domain (0.5 + 0.4 + 0.1 HTTPS) = 1.0
        item2 = {"url": "https://usa.gov", "claim_text": "claim1"}
        score2 = e2_source_reliability.calculate_reliability_score(item2, self.mock_trusted_sources, {})
        self.assertAlmostEqual(score2["score"], 1.0) # Cap at 1.0

        # 3. High trust (0.5 + 0.3 + 0.1 HTTPS) = 0.9
        item3 = {"url": "https://www.trusted.com/news", "claim_text": "claim1"}
        score3 = e2_source_reliability.calculate_reliability_score(item3, self.mock_trusted_sources, {})
        self.assertAlmostEqual(score3["score"], 0.9)

        # 4. Medium trust (0.5 + 0.1 + 0.0 HTTP) = 0.6
        item4 = {"url": "http://semi-trusted.com", "claim_text": "claim1"}
        score4 = e2_source_reliability.calculate_reliability_score(item4, self.mock_trusted_sources, {})
        self.assertAlmostEqual(score4["score"], 0.6)

    @patch('nodes.E_nodes.e2_source_reliability.load_trusted_sources')
    def test_consensus_boost(self, mock_load):
        mock_load.return_value = self.mock_trusted_sources
        
        # 3 sources for same claim -> Consensus boost (+0.1)
        # Base 0.5 + HTTPS 0.1 + Consensus 0.1 = 0.7
        
        evidence = [
            {"url": "https://a.com", "claim_text": "claim1"},
            {"url": "https://b.com", "claim_text": "claim1"},
            {"url": "https://c.com", "claim_text": "claim1"}
        ]
        
        state = {"evidence": evidence}
        
        # We need to mock check_about_page to avoid network calls
        with patch('nodes.E_nodes.e2_source_reliability.check_about_page', return_value=False):
            result_state = e2_source_reliability.run(state)
            
        for item in result_state["evidence"]:
            self.assertAlmostEqual(item["reliability_score"], 0.7)
            self.assertTrue(any("Consensus" in d for d in item["reliability_details"]))

    @patch('nodes.E_nodes.e2_source_reliability.check_about_page')
    def test_about_page_boost(self, mock_check_about):
        mock_check_about.return_value = True
        
        # Base 0.5 + HTTPS 0.1 + About 0.1 = 0.7
        item = {"url": "https://unknown.com", "claim_text": "claim1"}
        score = e2_source_reliability.calculate_reliability_score(item, self.mock_trusted_sources, {})
        
        self.assertAlmostEqual(score["score"], 0.7)
        self.assertIn("About page found (+0.1)", score["details"])

if __name__ == '__main__':
    unittest.main()
