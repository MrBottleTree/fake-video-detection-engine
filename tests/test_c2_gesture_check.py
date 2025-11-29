import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'nodes', 'C_nodes')))

from c2_gesture_narration_check import run

class TestC2GestureCheck(unittest.TestCase):
    @patch('c2_gesture_narration_check.SentenceTransformer')
    def test_clip_consistency_high(self, mock_transformer_cls):
        mock_model = MagicMock()
        mock_model.encode.side_effect = [
            np.array([[1.0, 0.0]]),
            np.array([[1.0, 0.0]])
        ]
        mock_transformer_cls.return_value = mock_model

        state = {
            "keyframes": ["/tmp/frame_000000.jpg"],
            "transcript": "A person is waving.",
            "segments": [{"start": 0.0, "end": 5.0, "text": "A person is waving."}],
            "metadata": {"video_fps": 1.0}
        }

        with patch('os.path.exists', return_value=True), \
             patch('PIL.Image.open', return_value=MagicMock()):
            
            result_state = run(state)
            
            self.assertEqual(len(result_state['gesture_check']), 1)
            self.assertEqual(result_state['gesture_check'][0]['status'], "Consistent")
            self.assertIn("CLIP Score", result_state['gesture_check'][0]['reason'])

    @patch('c2_gesture_narration_check.SentenceTransformer')
    @patch('c2_gesture_narration_check.OpenAI')
    @patch('os.getenv')
    def test_clip_ambiguous_openai_fallback(self, mock_getenv, mock_openai_cls, mock_transformer_cls):
        mock_getenv.return_value = "sk-dummy-key"

        mock_model = MagicMock()
        mock_model.encode.side_effect = [
            np.array([[1.0, 0.0]]), 
            np.array([[0.2, 0.979]]) 
        ]
        mock_transformer_cls.return_value = mock_model
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = '{"consistent": true, "reason": "OpenAI says yes"}'
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_cls.return_value = mock_client

        state = {
            "keyframes": ["/tmp/frame_000000.jpg"],
            "transcript": "Ambiguous action.",
            "segments": [{"start": 0.0, "end": 5.0, "text": "Ambiguous action."}],
            "metadata": {"video_fps": 1.0}
        }

        with patch('os.path.exists', return_value=True), \
             patch('PIL.Image.open', return_value=MagicMock()):
            
            result_state = run(state)
            
            self.assertEqual(len(result_state['gesture_check']), 1)
            self.assertEqual(result_state['gesture_check'][0]['status'], "Consistent")
            self.assertEqual(result_state['gesture_check'][0]['source'], "openai_fallback")

    def test_no_keyframes(self):
        state = {"keyframes": [], "transcript": "foo"}
        result_state = run(state)
        self.assertEqual(result_state, state)

if __name__ == '__main__':
    unittest.main()
