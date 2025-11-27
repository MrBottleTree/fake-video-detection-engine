import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add nodes directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'nodes', 'C_nodes')))

from c3_claim_extraction import run, extract_claims_openai

class TestC3OpenAIPrimary(unittest.TestCase):

    @patch('c3_claim_extraction.requests.post')
    def test_run_openai_primary_success(self, mock_post):
        """Test that OpenAI is called FIRST and used if successful."""
        # Mock successful OpenAI response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"claims": ["OpenAI Claim 1", "OpenAI Claim 2"]}'
                }
            }]
        }
        mock_post.return_value = mock_response

        state = {
            "transcript": "Some transcript text.",
            "ocr_results": []
        }
        
        result_state = run(state)
        
        # Verify claims come from OpenAI
        self.assertEqual(len(result_state['claims']), 2)
        self.assertEqual(result_state['claims'][0]['claim_text'], "OpenAI Claim 1")
        self.assertEqual(result_state['claims'][0]['source'], "openai_fallback") # Note: function still sets source as 'openai_fallback', maybe update?
        
        # Verify OpenAI was called
        mock_post.assert_called_once()

    @patch('c3_claim_extraction.get_spacy_model')
    @patch('c3_claim_extraction.extract_claims_openai')
    def test_run_openai_failure_fallback(self, mock_extract_openai, mock_get_spacy):
        """Test that if OpenAI fails, we fall back to SpaCy."""
        # Mock OpenAI failure (returns empty list)
        mock_extract_openai.return_value = []
        
        # Mock Spacy to return claims
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        
        # Create a mock sentence that passes is_claim_like
        mock_sent = MagicMock()
        mock_sent.text = "The sky is blue."
        mock_sent.ents = ["dummy_ent"] # Has entity -> high confidence
        # Mock token attributes for is_claim_like
        token1 = MagicMock(); token1.is_punct=False; token1.is_space=False; token1.pos_="VERB"; token1.dep_="ROOT"
        token2 = MagicMock(); token2.is_punct=False; token2.is_space=False; token2.pos_="NOUN"; token2.dep_="nsubj"
        # ... simplified mocking for is_claim_like is hard, let's just assume is_claim_like is imported and we can mock it or rely on it.
        # Actually, let's just mock the logic inside run loop by mocking doc.sents
        
        # Wait, is_claim_like is a separate function. I can mock it too if I want, or just construct a real-ish doc.
        # Let's mock the nlp call to return a doc with a sentence that is_claim_like returns True for.
        # But is_claim_like uses spacy token attributes.
        
        # Easier: Mock is_claim_like
        with patch('c3_claim_extraction.is_claim_like', return_value=True):
            mock_doc.sents = [mock_sent]
            mock_nlp.return_value = mock_doc
            mock_get_spacy.return_value = mock_nlp

            state = {
                "transcript": "The sky is blue.",
                "ocr_results": []
            }
            
            result_state = run(state)
            
            # Verify claims come from SpaCy (transcript_fallback)
            self.assertEqual(len(result_state['claims']), 1)
            self.assertEqual(result_state['claims'][0]['claim_text'], "The sky is blue.")
            self.assertEqual(result_state['claims'][0]['source'], "transcript_fallback")
            
            # Verify OpenAI was called
            mock_extract_openai.assert_called_once()

if __name__ == '__main__':
    unittest.main()
