import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.C_nodes.c3_claim_extraction import LLMClaimExtractor, run as run_c3
from nodes.E_nodes.e1_web_evidence import WebSearcher

class TestLLMIntegration(unittest.TestCase):
    
    def setUp(self):
        self.mock_state = {
            "transcript": "The inflation rate in 2022 was 8.5%.",
            "ocr_results": ["US Inflation 2022"],
            "debug": True
        }

    @patch('nodes.C_nodes.c3_claim_extraction.genai.GenerativeModel')
    def test_c3_llm_extraction(self, MockModel):
        """Test C3 uses LLM to extract claims."""
        # Mock LLM response
        mock_instance = MockModel.return_value
        mock_response = MagicMock()
        mock_response.text = json.dumps([
            {"claim_text": "The US inflation rate in 2022 was 8.5%", "confidence": 0.95, "source": "transcript+ocr"}
        ])
        mock_instance.generate_content.return_value = mock_response
        
        # Inject mock key to enable LLM
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_key"}):
            extractor = LLMClaimExtractor()
            claims = extractor.extract("transcript", "ocr")
            
            self.assertEqual(len(claims), 1)
            self.assertEqual(claims[0]["claim_text"], "The US inflation rate in 2022 was 8.5%")
            mock_instance.generate_content.assert_called_once()

    @patch('nodes.E_nodes.e1_web_evidence.genai.GenerativeModel')
    def test_e1_llm_query_generation(self, MockModel):
        """Test E1 uses LLM to generate queries."""
        # Mock LLM response
        mock_instance = MockModel.return_value
        mock_response = MagicMock()
        mock_response.text = json.dumps(["query1", "query2", "query3"])
        mock_instance.generate_content.return_value = mock_response
        
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_key"}):
            searcher = WebSearcher(debug=True)
            claim = {"id": "1", "claim_text": "Earth is flat", "who": None, "what": None, "when": None, "where": None}
            
            queries = searcher.construct_queries(claim)
            
            self.assertIn("query1", queries)
            self.assertEqual(len(queries), 3)
            mock_instance.generate_content.assert_called_once()

    def test_c3_fallback(self):
        """Test C3 falls back to Spacy when LLM fails/missing."""
        # Ensure no API key
        with patch.dict(os.environ, {"GOOGLE_API_KEY": ""}):
            # Use a longer transcript to ensure it passes the length heuristic (>5 tokens)
            fallback_state = {
                "transcript": "The quick brown fox jumps over the lazy dog and runs away.",
                "ocr_results": [],
                "debug": True
            }
            result_state = run_c3(fallback_state)
            
            print(f"DEBUG: Extracted claims: {result_state.get('claims')}")
            
            self.assertIn("claims", result_state)
            # Should have extracted something using Spacy fallback
            self.assertTrue(len(result_state["claims"]) > 0, "No claims extracted in fallback mode")
            self.assertIn("transcript_fallback", [c.get("source") for c in result_state["claims"]])

if __name__ == '__main__':
    unittest.main()
