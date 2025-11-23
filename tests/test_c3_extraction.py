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
            "transcript": "The sky is blue. Water is wet.",
            "ocr_results": []
        }
        result = c3_claim_extraction.run(state)
        claims = result["claims"]
        
        # "The sky is blue" -> 4 words (might be filtered if limit is >4)
        # "Water is wet" -> 3 words (filtered)
        # Let's check the logic in C3. 
        # Logic: len(tokens) < 4 returns False. So >= 4 is True.
        # "The sky is blue" is 4 tokens. Should be kept.
        
        texts = [c["claim_text"] for c in claims]
        self.assertIn("The sky is blue.", texts)
        self.assertNotIn("Water is wet.", texts)

    def test_spacy_segmentation(self):
        """Test that Spacy handles abbreviations correctly."""
        state = {
            "transcript": "Dr. Smith lives in the U.S.A. and he is a doctor.",
            "ocr_results": []
        }
        result = c3_claim_extraction.run(state)
        claims = result["claims"]
        
        # Should be one sentence: "Dr. Smith lives in the U.S.A. and he is a doctor."
        # Regex would likely split at "Dr." or "U.S.A."
        
        self.assertTrue(any("Dr. Smith lives in the U.S.A." in c["claim_text"] for c in claims))
        # Ensure it didn't split into "Dr."
        self.assertFalse(any(c["claim_text"] == "Dr." for c in claims))

    def test_filtering_heuristics(self):
        """Test filtering of questions and short sentences."""
        state = {
            "transcript": "Is the earth flat? No. The earth is definitely round.",
            "ocr_results": []
        }
        result = c3_claim_extraction.run(state)
        claims = result["claims"]
        
        texts = [c["claim_text"] for c in claims]
        
        # "Is the earth flat?" -> Question, filtered.
        self.assertNotIn("Is the earth flat?", texts)
        
        # "No." -> Too short, filtered.
        self.assertNotIn("No.", texts)
        
        # "The earth is definitely round." -> Valid.
        self.assertIn("The earth is definitely round.", texts)

    def test_ocr_integration(self):
        """Test that claims are extracted from OCR results."""
        state = {
            "transcript": "",
            "ocr_results": [
                {"text": "Breaking News: The economy is recovering fast."},
                "Simple string OCR result is here."
            ]
        }
        result = c3_claim_extraction.run(state)
        claims = result["claims"]
        
        texts = [c["claim_text"] for c in claims]
        self.assertIn("Breaking News: The economy is recovering fast.", texts)
        self.assertIn("Simple string OCR result is here.", texts)
        
        # Check source attribution
        for c in claims:
            self.assertEqual(c["source"], "ocr")

    def test_fallback_logic(self):
        """Test fallback to first sentence if no valid claims found."""
        # "Hi." is too short, so it would be filtered.
        # But fallback should pick it up if it's the only thing.
        state = {
            "transcript": "Hi there.",
            "ocr_results": []
        }
        result = c3_claim_extraction.run(state)
        claims = result["claims"]
        
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["source"], "transcript_fallback")
        self.assertIn("Hi there", claims[0]["claim_text"])

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
