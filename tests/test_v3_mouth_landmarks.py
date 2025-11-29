import unittest
import os
import sys
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.V_nodes import v3_mouth_landmarks_timeseries

import face_alignment

class TestV3MouthLandmarks(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_v3_landmarks"
        os.makedirs(self.test_dir, exist_ok=True)
        self.video_path = os.path.join(self.test_dir, "video.mp4")
        with open(self.video_path, "w") as f:
            f.write("dummy")

    def tearDown(self):
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch("cv2.VideoCapture")
    @patch("face_alignment.FaceAlignment")
    @patch("torch.cuda.is_available")
    @patch("cv2.VideoWriter")
    def test_landmark_extraction(self, mock_VideoWriter, mock_cuda_available, mock_FaceAlignment, mock_VideoCapture):
        print("\nTesting landmark extraction with Face Alignment...")
        
        mock_cuda_available.return_value = True
        
        mock_cap = MagicMock()
        mock_VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            5: 30.0,
            3: 640.0,
            4: 480.0,
            7: 100.0
        }.get(prop, 0.0)
        
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, dummy_frame), (False, None)]
        
        mock_fa_instance = MagicMock()
        mock_FaceAlignment.return_value = mock_fa_instance
        
        landmarks1 = np.zeros((68, 2), dtype=np.float32)
        landmarks1[48:68, :] = 10.0
        landmarks1[62] = [10, 10]
        landmarks1[66] = [10, 20]
        
        landmarks2 = np.zeros((68, 2), dtype=np.float32)
        landmarks2[48:68, :] = 50.0
        landmarks2[62] = [50, 50]
        landmarks2[66] = [50, 80]
        
        mock_fa_instance.get_landmarks.side_effect = [[landmarks1, landmarks2]]
        
        state = {
            "data_dir": self.test_dir,
            "debug": True
        }
        
        new_state = v3_mouth_landmarks_timeseries.run(state)
        
        mock_FaceAlignment.assert_called_with(face_alignment.LandmarksType.TWO_D, device='cuda', face_detector='sfd', face_detector_kwargs={'filter_threshold': 0.5})
        
        self.assertIn("mouth_landmarks", new_state)
        self.assertIn("mouth_landmarks_viz_path", new_state)
        landmarks_data = new_state["mouth_landmarks"]
        self.assertEqual(len(landmarks_data), 1)
        
        stored_landmarks = landmarks_data[0]["landmarks"]
        self.assertTrue(len(stored_landmarks) > 0)
        top_lip_stored = stored_landmarks[14]
        self.assertEqual(top_lip_stored, [50.0, 50.0])
        
        print("Landmark extraction test passed.")

    def test_missing_video(self):
        print("\nTesting missing video...")
        state = {"data_dir": "non_existent_dir"}
        new_state = v3_mouth_landmarks_timeseries.run(state)
        self.assertNotIn("mouth_landmarks", new_state)
        print("Missing video test passed.")

if __name__ == "__main__":
    unittest.main()
