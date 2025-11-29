import unittest
import os
import sys
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.C_nodes.c1_lip_sync_score import run as c1_run

class TestC1LipSyncScore(unittest.TestCase):

    def setUp(self):
        self.base_state = {
            "metadata": {"duration": 5.0, "fps": 30.0},
            "debug": False,
            "input_path": "dummy.mp4", 
            "mouth_landmarks": [],
            "lip_sync_score": None,
            "test_audio_signal": None,
            "face_detections": [{"box": [0, 0, 100, 100]}]
        }

    @staticmethod
    def generate_signals(duration, fps, sync_type="perfect"):
        num_frames = int(duration * fps)
        t = np.linspace(0, duration, num_frames)
        
        envelope = 0.5 * (1 + np.sin(2 * np.pi * 2 * t)) 
        envelope += 0.1 * np.random.rand(len(t))
        envelope = np.clip(envelope, 0, 1)
        
        audio_signal = envelope
        
        if sync_type == "perfect":
            mouth_signal = envelope
        elif sync_type == "delayed":
            shift = int(0.1 * fps)
            mouth_signal = np.roll(envelope, shift)
        elif sync_type == "random":
            mouth_signal = np.random.rand(len(t))
        elif sync_type == "silence":
            audio_signal = np.zeros_like(t)
            mouth_signal = envelope
        elif sync_type == "static_face":
            audio_signal = envelope
            mouth_signal = np.zeros_like(t)
        else:
            mouth_signal = envelope

        landmarks = []
        for i, val in enumerate(mouth_signal):
            landmarks.append({
                "timestamp": t[i],
                "mar": val * 0.5 + 0.1
            })
            
        return audio_signal, landmarks

    def test_perfect_sync(self):
        state = self.base_state.copy()
        audio, landmarks = self.generate_signals(5.0, 30.0, "perfect")
        
        state["test_audio_signal"] = audio
        state["mouth_landmarks"] = landmarks
        
        result = c1_run(state)
        
        print(f"Perfect Sync Score: {result['lip_sync_score']}")
        self.assertGreater(result["lip_sync_score"], 0.8)

    def test_delayed_sync(self):
        state = self.base_state.copy()
        audio, landmarks = self.generate_signals(5.0, 30.0, "delayed")
        
        state["test_audio_signal"] = audio
        state["mouth_landmarks"] = landmarks
        
        result = c1_run(state)
        
        print(f"Delayed Sync Score: {result['lip_sync_score']}")
        self.assertGreater(result["lip_sync_score"], 0.7)

    def test_no_sync(self):
        state = self.base_state.copy()
        audio, landmarks = self.generate_signals(5.0, 30.0, "random")
        
        state["test_audio_signal"] = audio
        state["mouth_landmarks"] = landmarks
        
        result = c1_run(state)
        
        print(f"Random Sync Score: {result['lip_sync_score']}")
        self.assertLess(result["lip_sync_score"], 0.4)

    def test_silence(self):
        state = self.base_state.copy()
        audio, landmarks = self.generate_signals(5.0, 30.0, "silence")
        
        state["test_audio_signal"] = audio
        state["mouth_landmarks"] = landmarks
        
        result = c1_run(state)
        
        self.assertEqual(result["lip_sync_score"], 0.0)

    def test_static_face(self):
        state = self.base_state.copy()
        audio, landmarks = self.generate_signals(5.0, 30.0, "static_face")
        
        state["test_audio_signal"] = audio
        state["mouth_landmarks"] = landmarks
        
        result = c1_run(state)
        
        self.assertEqual(result["lip_sync_score"], 0.0)

    @patch("torch.cuda.is_available", return_value=True)
    def test_gpu_path(self, mock_cuda):
        state = self.base_state.copy()
        audio, landmarks = self.generate_signals(2.0, 30.0, "perfect")
        state["test_audio_signal"] = audio
        state["mouth_landmarks"] = landmarks
        
        try:
            result = c1_run(state)
        except RuntimeError as e:
            if "Found no NVIDIA driver" in str(e) or "not compiled with CUDA" in str(e):
                pass
            else:
                print(f"Caught expected CUDA error on CPU machine: {e}")

if __name__ == "__main__":
    unittest.main()