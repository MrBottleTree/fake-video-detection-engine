import unittest
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TestOpenAIModelAccess(unittest.TestCase):
    """Test suite for OpenAI model access verification."""
    
    @unittest.skipIf(
        not os.getenv("OPENAI_API_KEY"),
        "Skipping model access test - OPENAI_API_KEY not found (GitHub Actions)"
    )
    def test_gpt4o_access(self):
        """Test access to gpt-4o model (used by C2, C3, V5 nodes)."""
        try:
            from openai import OpenAI
        except ImportError:
            self.skipTest("OpenAI package not installed")
        
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            self.assertIsNotNone(response)
        except Exception as e:
            self.fail(f"gpt-4o model not accessible: {e}")
    
    @unittest.skipIf(
        not os.getenv("OPENAI_API_KEY"),
        "Skipping model access test - OPENAI_API_KEY not found (GitHub Actions)"
    )
    def test_gpt4o_mini_access(self):
        """Test access to gpt-4o-mini model (fallback option)."""
        try:
            from openai import OpenAI
        except ImportError:
            self.skipTest("OpenAI package not installed")
        
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            self.assertIsNotNone(response)
        except Exception as e:
            self.fail(f"gpt-4o-mini model not accessible: {e}")
    
    @unittest.skipIf(
        not os.getenv("OPENAI_API_KEY"),
        "Skipping model access test - OPENAI_API_KEY not found (GitHub Actions)"
    )
    def test_gpt4_turbo_access(self):
        """Test access to gpt-4-turbo model."""
        try:
            from openai import OpenAI
        except ImportError:
            self.skipTest("OpenAI package not installed")
        
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            self.assertIsNotNone(response)
        except Exception as e:
            self.fail(f"gpt-4-turbo model not accessible: {e}")
    
    def test_openai_package_installed(self):
        """Test that openai package is installed (always runs)."""
        try:
            import openai
            self.assertTrue(True, "OpenAI package is installed")
        except ImportError:
            self.fail("OpenAI package is not installed")


if __name__ == '__main__':
    unittest.main()
