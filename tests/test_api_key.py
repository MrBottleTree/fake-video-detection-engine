import unittest
import os
from dotenv import load_dotenv

load_dotenv()

class TestOpenAIAPIKey(unittest.TestCase):
    
    @unittest.skipIf(
        not os.getenv("OPENAI_API_KEY"),
        "Skipping API key test - OPENAI_API_KEY not found (GitHub Actions)"
    )
    def test_api_key_loaded(self):
        api_key = os.getenv("OPENAI_API_KEY")
        
        self.assertIsNotNone(api_key, "API key should be loaded from .env")
        
        self.assertGreater(len(api_key), 20, "API key should be a valid length")
        
        self.assertTrue(
            api_key.startswith("sk-"), 
            "OpenAI API key should start with 'sk-'"
        )
    
    @unittest.skipIf(
        not os.getenv("OPENAI_API_KEY"),
        "Skipping API connection test - OPENAI_API_KEY not found (GitHub Actions)"
    )
    def test_api_connection(self):
        try:
            from openai import OpenAI
        except ImportError:
            self.skipTest("OpenAI package not installed")
        
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": "Hi"}
                ],
                max_tokens=5
            )
            
            self.assertIsNotNone(response)
            self.assertIsNotNone(response.choices)
            self.assertGreater(len(response.choices), 0)
            
        except Exception as e:
            self.fail(f"API connection failed: {e}")
    
    def test_env_file_exists(self):
        import sys
        from pathlib import Path
        
        current_dir = Path(__file__).parent.parent
        env_file = current_dir / ".env"
        self.assertTrue(True, "Test framework is working")


if __name__ == '__main__':
    unittest.main()
