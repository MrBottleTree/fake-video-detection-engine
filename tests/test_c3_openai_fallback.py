import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add nodes directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'nodes', 'C_nodes')))

from c3_claim_extraction import run, extract_claims_openai

class TestC3OpenAIFallback(unittest.TestCase):

    @patch('c3_claim_extraction.requests.post')
    def test_extract_claims_openai_success(self, mock_post):
        # Mock successful OpenAI response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"claims": ["The earth is round.", "Water boils at 100 degrees Celsius."]}'
                }
            }]
        }
        mock_post.return_value = mock_response

        transcript = "Some random text."
        ocr_results = ["Some text on screen"]
        
        claims = extract_claims_openai(transcript, ocr_results)
        
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0]['claim_text'], "The earth is round.")
        self.assertEqual(claims[0]['source'], "openai_fallback")
        self.assertEqual(claims[0]['confidence'], 0.95)

    @patch('c3_claim_extraction.requests.post')
    def test_extract_claims_openai_failure(self, mock_post):
        # Mock failed OpenAI response
        mock_post.side_effect = Exception("API Error")

        transcript = "Some random text."
        ocr_results = []
        
        claims = extract_claims_openai(transcript, ocr_results)
        
        self.assertEqual(len(claims), 0)

    @patch('c3_claim_extraction.get_spacy_model')
    @patch('c3_claim_extraction.extract_claims_openai')
    def test_run_fallback_trigger(self, mock_extract_openai, mock_get_spacy):
        # Mock Spacy to return no claims
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_doc.sents = [] # No sentences
        mock_nlp.return_value = mock_doc
        mock_get_spacy.return_value = mock_nlp
        
        # Mock OpenAI to return claims
        mock_extract_openai.return_value = [{
            "claim_text": "Fallback claim",
            "source": "openai_fallback",
            "confidence": 0.95
        }]

        state = {
            "transcript": "This is a transcript that spacy failed on.",
            "ocr_results": []
        }
        
        result_state = run(state)
        
        self.assertEqual(len(result_state['claims']), 1)
        self.assertEqual(result_state['claims'][0]['claim_text'], "Fallback claim")
        self.assertEqual(result_state['claims'][0]['source'], "openai_fallback")
        
        # Verify OpenAI was called
        mock_extract_openai.assert_called_once()

if __name__ == '__main__':
    unittest.main()
