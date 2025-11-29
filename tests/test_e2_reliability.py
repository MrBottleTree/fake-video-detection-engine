import unittest
from unittest.mock import patch, MagicMock, Mock
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nodes.E_nodes import e2_source_reliability

class TestE2Reliability(unittest.TestCase):

    def setUp(self):
        self.mock_trusted_sources = {
            "high_trust": ["trusted.com"],
            "medium_trust": ["semi-trusted.com"]
        }
        
    def tearDown(self):
        pass

    def test_get_domain(self):
        self.assertEqual(e2_source_reliability.get_domain("https://www.example.com/page"), "example.com")
        self.assertEqual(e2_source_reliability.get_domain("http://sub.domain.org"), "sub.domain.org")
        self.assertEqual(e2_source_reliability.get_domain("invalid-url"), "")

    @patch('nodes.E_nodes.e2_source_reliability.openai_client', None)
    @patch('nodes.E_nodes.e2_source_reliability.load_trusted_sources')
    @patch('urllib.request.urlopen')
    def test_calculate_reliability_score_heuristic(self, mock_urlopen, mock_load):
        mock_load.return_value = self.mock_trusted_sources
        mock_urlopen.side_effect = Exception("Network error")

        item1 = {"url": "https://unknown.com", "claim_text": "claim1", "snippet": "test"}
        score1 = e2_source_reliability.calculate_reliability_score(item1, self.mock_trusted_sources, {})
        self.assertAlmostEqual(score1["score"], 0.6)
        item2 = {"url": "https://usa.gov", "claim_text": "claim1", "snippet": "test"}
        score2 = e2_source_reliability.calculate_reliability_score(item2, self.mock_trusted_sources, {})
        self.assertAlmostEqual(score2["score"], 1.0)  # Cap at 1.0
        item3 = {"url": "https://www.trusted.com/news", "claim_text": "claim1", "snippet": "test"}
        score3 = e2_source_reliability.calculate_reliability_score(item3, self.mock_trusted_sources, {})
        self.assertAlmostEqual(score3["score"], 0.9)
        item4 = {"url": "http://semi-trusted.com", "claim_text": "claim1", "snippet": "test"}
        score4 = e2_source_reliability.calculate_reliability_score(item4, self.mock_trusted_sources, {})
        self.assertAlmostEqual(score4["score"], 0.6)

    @patch('nodes.E_nodes.e2_source_reliability.load_trusted_sources')
    def test_openai_evaluation_success(self, mock_load):
        mock_load.return_value = self.mock_trusted_sources
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "score": 0.85,
            "reason": "This is a reputable source"
        })
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('nodes.E_nodes.e2_source_reliability.openai_client', mock_client):
            item = {
                "url": "https://example.com",
                "claim_text": "The ocean has a lot of water",
                "snippet": "The ocean is a large body of water"
            }
            score = e2_source_reliability.calculate_reliability_score(item, self.mock_trusted_sources, {})
            
            self.assertAlmostEqual(score["score"], 0.85)
            self.assertTrue(any("OpenAI" in d for d in score["details"]))
            
    @patch('nodes.E_nodes.e2_source_reliability.load_trusted_sources')
    def test_openai_evaluation_failure_fallback(self, mock_load):
        mock_load.return_value = self.mock_trusted_sources
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        with patch('nodes.E_nodes.e2_source_reliability.openai_client', mock_client):
            with patch('urllib.request.urlopen', side_effect=Exception("Network error")):
                item = {
                    "url": "https://unknown.com",
                    "claim_text": "test claim",
                    "snippet": "test snippet"
                }
                score = e2_source_reliability.calculate_reliability_score(item, self.mock_trusted_sources, {})
                
                self.assertAlmostEqual(score["score"], 0.6)
                self.assertFalse(any("OpenAI" in d for d in score["details"]))

    @patch('nodes.E_nodes.e2_source_reliability.openai_client', None)
    @patch('nodes.E_nodes.e2_source_reliability.load_trusted_sources')
    def test_consensus_boost(self, mock_load):
        mock_load.return_value = self.mock_trusted_sources
        
        evidence = [
            {"url": "https://a.com", "claim_text": "claim1", "snippet": "test"},
            {"url": "https://b.com", "claim_text": "claim1", "snippet": "test"},
            {"url": "https://c.com", "claim_text": "claim1", "snippet": "test"}
        ]
        
        state = {"evidence": evidence}
        
        with patch('nodes.E_nodes.e2_source_reliability.check_about_page', return_value=False):
            result_state = e2_source_reliability.run(state)
            
        for item in result_state["evidence"]:
            self.assertAlmostEqual(item["reliability_score"], 0.7)
            self.assertTrue(any("Consensus" in d for d in item["reliability_details"]))

    @patch('nodes.E_nodes.e2_source_reliability.openai_client', None)
    @patch('nodes.E_nodes.e2_source_reliability.check_about_page')
    def test_about_page_boost(self, mock_check_about):
        mock_check_about.return_value = True
        
        item = {"url": "https://unknown.com", "claim_text": "claim1", "snippet": "test"}
        score = e2_source_reliability.calculate_reliability_score(item, self.mock_trusted_sources, {})
        
        self.assertAlmostEqual(score["score"], 0.7)
        self.assertIn("About page found (+0.1)", score["details"])

if __name__ == '__main__':
    unittest.main()
