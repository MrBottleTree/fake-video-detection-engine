import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add nodes directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'nodes', 'C_nodes')))

from c3_claim_extraction import run, extract_claims_openai

class TestC3OpenAIFallback(unittest.TestCase):

    @patch('c3_claim_extraction.OpenAI')
    def test_extract_claims_openai_success(self, mock_openai_class):
        # Mock successful OpenAI response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"claims": ["The earth is round.", "Water boils at 100 degrees Celsius." ]}'
        mock_response.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        with patch('c3_claim_extraction.OPENAI_API_KEY', 'test_key'):
            transcript = "Some random text."
            ocr_results = ["Some text on screen"]
            
            claims = extract_claims_openai(transcript, ocr_results)
            
            self.assertEqual(len(claims), 2)
            self.assertEqual(claims[0]['claim_text'], "The earth is round.")
            self.assertEqual(claims[0]['source'], "openai")
            self.assertEqual(claims[0]['confidence'], 0.95)

    @patch('c3_claim_extraction.OpenAI')
    def test_extract_claims_openai_failure(self, mock_openai_class):
        # Mock failed OpenAI response
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client

        with patch('c3_claim_extraction.OPENAI_API_KEY', 'test_key'):
            transcript = "Some random text."
            ocr_results = []
            
            claims = extract_claims_openai(transcript, ocr_results)
            
            self.assertEqual(len(claims), 0)

    # Fallback trigger test removed as fallback logic is deprecated.

if __name__ == '__main__':
    unittest.main()
