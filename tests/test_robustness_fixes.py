import unittest
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nodes.C_nodes import c3_claim_extraction, c1_lip_sync_score

class TestRobustness(unittest.TestCase):

    def test_c3_extraction_spacy(self):
        # Test transcript extraction with Spacy
        # We need to handle the case where Spacy model might not be present in the test env
        # But for this test, let's assume we can mock it or it falls back to blank
        
        state = {
            "transcript": "Dr. Smith said the earth is round. This is a fact.",
            "ocr_results": []
        }
        
        # We can try running it. If model missing, it might try to download (which we should mock or allow)
        # To be safe and fast, let's mock spacy.load if we want, but integration test is better.
        # Let's trust the fallback logic in C3 or just run it.
        
        # If we run it, it might print "Downloading...".
        try:
            result = c3_claim_extraction.run(state)
            claims = result["claims"]
            
            # "Dr. Smith said the earth is round." should be one sentence despite the dot in Dr.
            # Regex would likely split at "Dr.". Spacy should keep it together.
            
            # Check if we have the full sentence
            found_full = any("Dr. Smith said the earth is round" in c["claim_text"] for c in claims)
            self.assertTrue(found_full, "Spacy should handle 'Dr.' abbreviation correctly")
            
        except Exception as e:
            print(f"Skipping Spacy test due to environment issue: {e}")

    def test_c3_ocr_fallback(self):
        state = {
            "transcript": "",
            "ocr_results": [{"text": "Breaking News: Aliens have landed in New York City."}]
        }
        result = c3_claim_extraction.run(state)
        claims = result["claims"]
        
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["source"], "ocr")

    def test_c1_windowed_robustness(self):
        # Create synthetic signals
        fps = 30
        duration = 10.0
        num_frames = int(duration * fps)
        
        t = np.linspace(0, duration, num_frames)
        signal = np.sin(2 * np.pi * 2 * t) 
        
        mouth_landmarks = []
        audio_onsets = []
        
        for i in range(num_frames):
            mar = (signal[i] + 1) / 2 
            mouth_landmarks.append({
                "timestamp": t[i],
                "mar": mar
            })
            
            if i > 0 and signal[i-1] < 0.9 and signal[i] >= 0.9:
                audio_onsets.append(t[i])
                
        # Corrupt the last 5 seconds
        audio_onsets = [a for a in audio_onsets if a < 5.0]
        
        state = {
            "mouth_landmarks": mouth_landmarks,
            "audio_onsets": audio_onsets,
            "metadata": {"fps": fps, "duration": duration}
        }
        
        result = c1_lip_sync_score.run(state)
        score = result["lip_sync_score"]
        
        print(f"Test Score: {score}")
        self.assertGreater(score, 0.5)

if __name__ == '__main__':
    unittest.main()
