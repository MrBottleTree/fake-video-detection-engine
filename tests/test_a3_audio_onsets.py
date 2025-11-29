import unittest
import os
import sys
import numpy as np
import librosa
from scipy.io import wavfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.A_nodes import a3_audio_onsets

class TestA3AudioOnsets(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_audio_onsets"
        os.makedirs(self.test_dir, exist_ok=True)
        self.test_audio = os.path.join(self.test_dir, "audio_16k.wav")
        
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        audio = np.zeros_like(t)
        
        idx1 = int(0.5 * sr)
        audio[idx1:idx1+1000] = np.sin(2 * np.pi * 440 * t[idx1:idx1+1000])
        
        idx2 = int(1.5 * sr)
        audio[idx2:idx2+1000] = np.sin(2 * np.pi * 880 * t[idx2:idx2+1000])
        
        wavfile.write(self.test_audio, sr, audio.astype(np.float32))

    def tearDown(self):
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_onset_detection(self):
        print("\nTesting onset detection...")
        state = {
            "data_dir": self.test_dir,
            "metadata": {"fps": 30.0, "duration": 2.0}
        }
        
        new_state = a3_audio_onsets.run(state)
        
        onsets = new_state.get("audio_onsets")
        self.assertIsNotNone(onsets)
        self.assertTrue(len(onsets) >= 2)
        
        has_onset_1 = any(abs(o - 0.5) < 0.1 for o in onsets)
        has_onset_2 = any(abs(o - 1.5) < 0.1 for o in onsets)
        
        self.assertTrue(has_onset_1, "Did not detect onset at 0.5s")
        self.assertTrue(has_onset_2, "Did not detect onset at 1.5s")
        
        envelope = new_state.get("audio_envelope")
        self.assertIsNotNone(envelope)
        self.assertEqual(len(envelope), int(2.0 * 30.0))
        self.assertGreater(envelope[15], 0.1)
        self.assertGreater(envelope[45], 0.1)
        
        self.assertEqual(new_state["metadata"]["onset_detection_method"], "librosa.onset.onset_detect")
        print("Onset detection test passed.")

    def test_missing_audio(self):
        print("\nTesting missing audio file...")
        state = {"data_dir": "non_existent_dir"}
        new_state = a3_audio_onsets.run(state)
        self.assertNotIn("audio_onsets", new_state)
        print("Missing audio test passed.")

if __name__ == "__main__":
    unittest.main()
