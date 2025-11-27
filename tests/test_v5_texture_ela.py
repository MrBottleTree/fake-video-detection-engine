import unittest
import os
import sys
import numpy as np
import cv2
import shutil
import tempfile
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nodes.V_nodes.v5_texture_ela import run

class TestV5TextureELA(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.face_dir = os.path.join(self.test_dir, "faces")
        os.makedirs(self.face_dir)
        
        # Create a dummy face image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (25, 25), (75, 75), (255, 255, 255), -1)
        
        self.face_path = os.path.join(self.face_dir, "face_0.jpg")
        cv2.imwrite(self.face_path, img)
        
        self.mock_state = {
            "face_detections": [
                {
                    "faces": [
                        {
                            "confidence": 0.99,
                            "bbox": {"w": 100, "h": 100},
                            "crop_path": self.face_path
                        }
                    ]
                }
            ],
            "data_dir": self.test_dir,
            "debug": True
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_v5_no_faces(self):
        state = {"face_detections": [], "data_dir": "dummy"}
        result = run(state)
        self.assertEqual(result["texture_ela_score"], 0.0)
        self.assertEqual(result["texture_ela_details"]["reason"], "No faces found")

    @patch("nodes.V_nodes.v5_texture_ela.OpenAI")
    def test_v5_run_basic(self, mock_openai):
        # Mock OpenAI response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"fake_probability": 0.85, "reasoning": "Test reasoning"}'
        mock_client.chat.completions.create.return_value = mock_response
        
        result = run(self.mock_state)
        
        self.assertIn("texture_ela_score", result)
        self.assertEqual(result["texture_ela_score"], 0.85)
        self.assertEqual(len(result["texture_ela_details"]), 1)
        self.assertEqual(result["texture_ela_details"][0]["fake_probability"], 0.85)
        
        # Verify ELA and FFT images were created
        ela_dir = os.path.join(self.mock_state["data_dir"], "ela_analysis")
        self.assertTrue(os.path.exists(os.path.join(ela_dir, "ela_0.jpg")))
        self.assertTrue(os.path.exists(os.path.join(ela_dir, "fft_0.jpg")))

    @patch("nodes.V_nodes.v5_texture_ela.OpenAI")
    def test_v5_openai_failure(self, mock_openai):
        # Mock OpenAI to raise an exception
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        result = run(self.mock_state)
        
        # Should handle error gracefully
        self.assertEqual(result["texture_ela_score"], 0.0)
        self.assertEqual(result["texture_ela_details"]["reason"], "Analysis failed or no keys")

if __name__ == "__main__":
    unittest.main()
