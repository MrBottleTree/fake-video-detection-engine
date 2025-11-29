import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'nodes', 'C_nodes')))

from c3_claim_extraction import run, extract_claims_openai

class TestC3OpenAIPrimary(unittest.TestCase):

    @patch('c3_claim_extraction.OpenAI')
    def test_run_openai_primary_success(self, mock_openai_class):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"claims": ["OpenAI Claim 1", "OpenAI Claim 2"]}'
        mock_response.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            with patch('c3_claim_extraction.OPENAI_API_KEY', 'test_key'):
                 state = {
                    "transcript": "Some transcript text.",
                    "ocr_results": []
                }
                
                 result_state = run(state)
        
        self.assertEqual(len(result_state['claims']), 2)
        self.assertEqual(result_state['claims'][0]['claim_text'], "OpenAI Claim 1")
        self.assertEqual(result_state['claims'][0]['source'], "openai")
        
        mock_client.chat.completions.create.assert_called_once()


if __name__ == '__main__':
    unittest.main()
