import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nodes.C_nodes import c3_claim_extraction

class TestC3Extraction(unittest.TestCase):

    def setUp(self):
        # Reset the global model cache before tests if needed, 
        # but usually we want to keep it to save time.
        pass

    def test_basic_extraction(self):
        """Test extraction of simple valid claims."""
        state = {
            "transcript": "The sky is blue and beautiful. Water is wet.",
            "ocr_results": []
        }
        result = c3_claim_extraction.run(state)
        claims = result["claims"]
        
        # "The sky is blue and beautiful." -> >5 tokens, has subj+verb. Should be kept.
        # "Water is wet." -> 3 tokens (filtered).
        
        texts = [c["claim_text"] for c in claims]
        self.assertTrue(any("The sky is blue and beautiful" in t for t in texts))
        self.assertFalse(any("Water is wet" in t for t in texts))

    def test_spacy_segmentation(self):
        """Test that Spacy handles abbreviations correctly."""
        state = {
            "transcript": "Dr. Smith lives in the U.S.A. and he is a doctor.",
            "ocr_results": []
        }
        result = c3_claim_extraction.run(state)
        claims = result["claims"]
        
        # Should be one sentence: "Dr. Smith lives in the U.S.A. and he is a doctor."
        self.assertTrue(any("Dr. Smith lives in the U.S.A." in c["claim_text"] for c in claims))
        self.assertFalse(any(c["claim_text"] == "Dr." for c in claims))

    def test_filtering_heuristics(self):
        """Test filtering of questions and short sentences."""
        state = {
            "transcript": "Is the earth flat? No. The earth is definitely round and orbits the sun.",
            "ocr_results": []
        }
        result = c3_claim_extraction.run(state)
        claims = result["claims"]
        
        texts = [c["claim_text"] for c in claims]
        
        # "Is the earth flat?" -> Question, filtered.
        self.assertFalse(any("Is the earth flat" in t for t in texts))
        
        # "No." -> Too short, filtered.
        self.assertNotIn("No.", texts)
        
        # "The earth is definitely round and orbits the sun." -> Valid.
        self.assertTrue(any("The earth is definitely round" in t for t in texts))

    def test_ocr_integration(self):
        """Test that claims are extracted from OCR results."""
        state = {
            "transcript": "",
            "ocr_results": [
                {"text": "Breaking News: The economy is recovering fast from the recession."},
                "Simple string OCR result is here."
            ]
        }
        result = c3_claim_extraction.run(state)
        claims = result["claims"]
        
        texts = [c["claim_text"] for c in claims]
        self.assertTrue(any("The economy is recovering fast" in t for t in texts))
        # "Simple string OCR result is here." -> 6 tokens. might pass if structure is ok.
        
        # Check source attribution
        for c in claims:
            self.assertEqual(c["source"], "ocr")

    def test_fallback_logic(self):
        """Test fallback to first sentence if no valid claims found."""
        # "Hi there." is too short, so it would be filtered.
        # But fallback should pick it up if it's the only thing.
        state = {
            "transcript": "Hi there my friend.",
            "ocr_results": []
        }
        result = c3_claim_extraction.run(state)
        claims = result["claims"]
        
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["source"], "transcript_fallback")
        self.assertIn("Hi there my friend", claims[0]["claim_text"])

    def test_deduplication(self):
        """Test that duplicate claims are removed."""
        state = {
            "transcript": "The earth is round. The earth is round.",
            "ocr_results": []
        }
        result = c3_claim_extraction.run(state)
        claims = result["claims"]
        
        self.assertEqual(len(claims), 1)

if __name__ == '__main__':
    unittest.main()
