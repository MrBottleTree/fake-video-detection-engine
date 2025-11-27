import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add nodes directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'nodes', 'C_nodes')))

from c3_claim_extraction import run, extract_claims_openai

class TestC3OpenAIPrimary(unittest.TestCase):

    @patch('c3_claim_extraction.OpenAI')
    def test_run_openai_primary_success(self, mock_openai_class):
        """Test that OpenAI is called FIRST and used if successful."""
        # Mock successful OpenAI response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"claims": ["OpenAI Claim 1", "OpenAI Claim 2"]}'
        mock_response.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # Ensure API key is set for the test
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            # Reload module to pick up env var if needed, but here we patch os.getenv or just rely on the fact that run() calls extract which checks env.
            # Actually, the module level OPENAI_API_KEY is loaded at import time.
            # So we might need to patch the module variable.
            with patch('c3_claim_extraction.OPENAI_API_KEY', 'test_key'):
                 state = {
                    "transcript": "Some transcript text.",
                    "ocr_results": []
                }
                
                 result_state = run(state)
        
        # Verify claims come from OpenAI
        self.assertEqual(len(result_state['claims']), 2)
        self.assertEqual(result_state['claims'][0]['claim_text'], "OpenAI Claim 1")
        self.assertEqual(result_state['claims'][0]['source'], "openai")
        
        # Verify OpenAI was called
        mock_client.chat.completions.create.assert_called_once()

    # Fallback test removed as fallback logic is deprecated.

if __name__ == '__main__':
    unittest.main()
