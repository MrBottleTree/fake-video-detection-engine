import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.E_nodes.e1_web_evidence import run, WebSearcher

class TestE1Robustness(unittest.TestCase):
    
    def setUp(self):
        self.mock_state = {
            "claims": [
                {"claim_text": "Python is a programming language.", "who": "Guido", "what": "Python", "when": "1991", "where": "Netherlands"},
                "Simple string claim"
            ],
            "debug": True,
            "use_cache": False
        }

    @patch('nodes.E_nodes.e1_web_evidence.WebSearcher')
    def test_e1_run_structure(self, MockSearcher):
        """Test that E1 run function returns state with evidence key and IDs."""
        # Mock searcher instance
        instance = MockSearcher.return_value
        instance.construct_queries.return_value = ["query1"]
        instance.search_robust.return_value = [
            {"url": "http://test.com", "title": "Test", "snippet": "Snippet", "source": "test", "date": None, "relevance_score": 0.9}
        ]
        instance.deduplicate.return_value = instance.search_robust.return_value
        instance.rank_results.return_value = instance.search_robust.return_value

        result_state = run(self.mock_state)
        
        self.assertIn("evidence", result_state)
        # Should now have 2 evidence items (1 per claim, since we mocked 1 search result per claim)
        self.assertEqual(len(result_state["evidence"]), 2)
        
        # Check flattened structure - each evidence item should have claim_text and claim_id
        first_item = result_state["evidence"][0]
        self.assertIn("claim_id", first_item)
        self.assertIn("claim_text", first_item)
        self.assertEqual(first_item["claim_text"], "Python is a programming language.")
        self.assertIn("url", first_item)
        self.assertEqual(first_item["url"], "http://test.com")

    def test_smart_queries(self):
        """Test that smart queries (debunking/supporting) are generated."""
        searcher = WebSearcher(debug=True, use_cache=False)
        claim = {
            "id": "123",
            "claim_text": "Earth is flat",
            "who": None, "what": None, "when": None, "where": None
        }
        
        queries = searcher.construct_queries(claim)
        
        # Check for variety
        self.assertTrue(any("debunked" in q for q in queries))
        self.assertTrue(any("proof" in q for q in queries))
        self.assertTrue(any("fact check" in q for q in queries))

    def test_web_searcher_fallback(self):
        """Test that WebSearcher falls back to different APIs."""
        searcher = WebSearcher(debug=True, use_cache=False)
        
        # Mock all search methods
        searcher._search_serper = MagicMock(side_effect=Exception("Serper failed"))
        searcher._search_google = MagicMock(side_effect=Exception("Google failed"))
        searcher._search_ddg = MagicMock(return_value=[{"url": "ddg.com", "title": "DDG", "snippet": "DDG", "source": "ddg", "date": None, "relevance_score": 0.0}])
        
        # Set keys to trigger attempts
        searcher.serper_key = "fake_key"
        searcher.google_key = "fake_key"
        searcher.google_cx = "fake_cx"
        
        results = searcher.search_robust("test query")
        
        # Verify fallback chain
        searcher._search_serper.assert_called_once()
        searcher._search_google.assert_called_once()
        searcher._search_ddg.assert_called_once()
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source"], "ddg")

if __name__ == '__main__':
    unittest.main()
