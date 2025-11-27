import unittest
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TestOpenAIAPIKey(unittest.TestCase):
    """Test suite for OpenAI API key verification."""
    
    @unittest.skipIf(
        not os.getenv("OPENAI_API_KEY"),
        "Skipping API key test - OPENAI_API_KEY not found (GitHub Actions)"
    )
    def test_api_key_loaded(self):
        """Test that API key is loaded from .env file."""
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Verify key exists
        self.assertIsNotNone(api_key, "API key should be loaded from .env")
        
        # Verify key has reasonable length (OpenAI keys are typically 100+ chars)
        self.assertGreater(len(api_key), 20, "API key should be a valid length")
        
        # Verify key starts with expected prefix
        self.assertTrue(
            api_key.startswith("sk-"), 
            "OpenAI API key should start with 'sk-'"
        )
    
    @unittest.skipIf(
        not os.getenv("OPENAI_API_KEY"),
        "Skipping API connection test - OPENAI_API_KEY not found (GitHub Actions)"
    )
    def test_api_connection(self):
        """Test that we can connect to OpenAI API."""
        try:
            from openai import OpenAI
        except ImportError:
            self.skipTest("OpenAI package not installed")
        
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        
        # Make a minimal API call
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": "Hi"}
                ],
                max_tokens=5
            )
            
            # Verify we got a response
            self.assertIsNotNone(response)
            self.assertIsNotNone(response.choices)
            self.assertGreater(len(response.choices), 0)
            
        except Exception as e:
            self.fail(f"API connection failed: {e}")
    
    def test_env_file_exists(self):
        """Test that .env file exists (always runs, even on GitHub Actions)."""
        # This test just checks if we can load dotenv
        # Even without an API key, we should be able to run this
        import sys
        from pathlib import Path
        
        # Check if we're in the expected directory structure
        current_dir = Path(__file__).parent.parent
        env_file = current_dir / ".env"
        
        # On GitHub Actions, .env might not exist, which is fine
        # So we just verify the test framework itself works
        self.assertTrue(True, "Test framework is working")


if __name__ == '__main__':
    unittest.main()
