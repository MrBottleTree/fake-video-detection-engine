import unittest
import os
import sys
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.V_nodes import v3_mouth_landmarks_timeseries

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
    @patch("cv2.VideoWriter")
    @patch("mediapipe.solutions.drawing_utils.draw_landmarks")
    @patch("mediapipe.solutions.face_mesh.FaceMesh")
    def test_landmark_extraction(self, mock_FaceMesh, mock_draw_landmarks, mock_VideoWriter, mock_VideoCapture):
        print("\nTesting MediaPipe landmark extraction...")
        
        mock_cap = MagicMock()
        mock_VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            5: 30.0,
            7: 10.0,
            3: 640.0,
            4: 480.0
        }.get(prop, 0.0)
        
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, dummy_frame), (False, None)]
        
        mock_face_mesh_instance = MagicMock()
        mock_FaceMesh.return_value = mock_face_mesh_instance
        
        mock_face1 = MagicMock()
        mock_landmarks1 = []
        for i in range(478):
            mock_landmark = MagicMock()
            if i == 13:
                mock_landmark.x = 0.5
                mock_landmark.y = 0.4
            elif i == 14:
                mock_landmark.x = 0.5
                mock_landmark.y = 0.45
            else:
                mock_landmark.x = 0.5
                mock_landmark.y = 0.5
            mock_landmark.z = 0.0
            mock_landmark.visibility = 1.0
            mock_landmark.presence = 1.0
            mock_landmarks1.append(mock_landmark)
        
        type(mock_face1).landmark = PropertyMock(return_value=mock_landmarks1)
        
        mock_face2 = MagicMock()
        mock_landmarks2 = []
        for i in range(478):
            mock_landmark = MagicMock()
            if i == 13:
                mock_landmark.x = 0.5
                mock_landmark.y = 0.3
            elif i == 14:
                mock_landmark.x = 0.5
                mock_landmark.y = 0.5
            else:
                mock_landmark.x = 0.5
                mock_landmark.y = 0.5
            mock_landmark.z = 0.0
            mock_landmark.visibility = 1.0
            mock_landmark.presence = 1.0
            mock_landmarks2.append(mock_landmark)
        
        type(mock_face2).landmark = PropertyMock(return_value=mock_landmarks2)
        
        mock_results = MagicMock()
        mock_results.multi_face_landmarks = [mock_face1, mock_face2]
        mock_face_mesh_instance.process.return_value = mock_results
        
        state = {
            "data_dir": self.test_dir,
            "debug": True
        }
        
        new_state = v3_mouth_landmarks_timeseries.run(state)
        
        self.assertIn("mouth_landmarks", new_state)
        self.assertIn("mouth_landmarks_viz_path", new_state)
        
        landmarks_data = new_state["mouth_landmarks"]
        self.assertEqual(len(landmarks_data), 1)
        
        frame_data = landmarks_data[0]
        self.assertGreater(frame_data["lip_distance"], 90)
        self.assertLess(frame_data["lip_distance"], 100)
        
        self.assertTrue(len(frame_data["landmarks"]) > 0)
        self.assertEqual(new_state["metadata"]["landmark_model"], "mediapipe_face_mesh")
        self.assertEqual(new_state["metadata"]["landmark_count"], 40)
        
        print("MediaPipe landmark extraction test passed.")

    def test_missing_video(self):
        print("\nTesting missing video...")
        state = {"data_dir": "non_existent_dir"}
        new_state = v3_mouth_landmarks_timeseries.run(state)
        self.assertNotIn("mouth_landmarks", new_state)
        print("Missing video test passed.")

if __name__ == "__main__":
    unittest.main()
