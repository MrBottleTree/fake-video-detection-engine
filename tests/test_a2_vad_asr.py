import unittest
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.A_nodes import a2_vad_asr

class TestA2VadAsr(unittest.TestCase):
    def setUp(self):
        self.test_audio = "test_audio.wav"
        with open(self.test_audio, "w") as f:
            f.write("dummy audio content")

    def tearDown(self):
        if os.path.exists(self.test_audio):
            os.remove(self.test_audio)

    @patch('nodes.A_nodes.a2_vad_asr.whisper')
    @patch('nodes.A_nodes.a2_vad_asr.torch')
    def test_vad_asr_success(self, mock_torch, mock_whisper):
        print("\nTesting VAD and ASR success...")
        
        mock_torch.cuda.is_available.return_value = False
        
        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model
        
        mock_result = {
            "text": "Hello world this is a test.",
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Hello world"},
                {"start": 2.0, "end": 4.0, "text": "this is a test."}
            ]
        }
        mock_model.transcribe.return_value = mock_result
        
        state = {"audio_path": self.test_audio}
        
        new_state = a2_vad_asr.run(state)
        
        mock_whisper.load_model.assert_called_with("base", device="cpu")
        mock_model.transcribe.assert_called_with(self.test_audio)
        
        self.assertEqual(new_state["transcript"], "Hello world this is a test.")
        self.assertEqual(len(new_state["segments"]), 2)
        self.assertEqual(new_state["word_count"], 6)
        self.assertEqual(new_state["metadata"]["transcription_model"], "openai-whisper-base")
        
        print("VAD and ASR success test passed.")

    def test_missing_audio(self):
        print("\nTesting missing audio file...")
        state = {"audio_path": "non_existent.wav"}
        new_state = a2_vad_asr.run(state)
        self.assertNotIn("transcript", new_state)
        print("Missing audio test passed.")

if __name__ == "__main__":
    unittest.main()
