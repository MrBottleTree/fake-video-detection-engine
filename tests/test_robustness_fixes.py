import unittest
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nodes.C_nodes import c3_claim_extraction, c1_lip_sync_score

class TestRobustness(unittest.TestCase):

    def test_c1_windowed_robustness(self):
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
        audio_signal = (signal + 1) / 2
        half_frames = num_frames // 2
        audio_signal[half_frames:] = 0.0
        
        state = {
            "mouth_landmarks": mouth_landmarks,
            "audio_onsets": audio_onsets,
            "test_audio_signal": audio_signal.tolist(),
            "metadata": {"fps": fps, "duration": duration},
            "face_detections": [{"box": [0, 0, 100, 100]}]
        }
        
        result = c1_lip_sync_score.run(state)
        score = result["lip_sync_score"]
        
        print(f"Test Score: {score}")
        self.assertGreater(score, 0.5)

if __name__ == '__main__':
    unittest.main()
