import unittest
import os
import shutil
import numpy as np
from unittest.mock import MagicMock, patch
from nodes.V_nodes import v4_blink_headpose_dynamics

class TestV4BlinkHeadPose(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_v4_data"
        os.makedirs(self.test_dir, exist_ok=True)
        self.video_path = os.path.join(self.test_dir, "video.mp4")
        
        with open(self.video_path, "wb") as f:
            f.write(b"dummy video content")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch("cv2.VideoCapture")
    @patch("face_alignment.FaceAlignment")
    @patch("nodes.V_nodes.v4_blink_headpose_dynamics.SixDRepNet")
    @patch("cv2.VideoWriter")
    def test_blink_and_pose_extraction(self, mock_VideoWriter, mock_SixDRepNet, mock_FaceAlignment, mock_VideoCapture):
        print("\nTesting V4 blink and pose extraction...")

        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            5: 30.0,  # FPS
            3: 640.0, # Width
            4: 480.0, # Height
            7: 100.0  # Frame Count
        }.get(prop, 0.0)

        # Mock frames
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, dummy_frame), (True, dummy_frame), (False, None)]

        # Mock FaceAlignment
        mock_fa = MagicMock()
        mock_FaceAlignment.return_value = mock_fa
        
        # Create dummy landmarks (68 points, 3D)
        # We need specific points for eyes and nose to test EAR and Pose
        landmarks = np.zeros((68, 3), dtype=np.float32)
        
        # Set Nose (30)
        landmarks[30] = [320, 240, 0]
        # Set Chin (8)
        landmarks[8] = [320, 340, 0]
        # Set Eyes (36-41, 42-47)
        # Open eyes
        landmarks[36] = [280, 200, 0]; landmarks[39] = [300, 200, 0]
        landmarks[37] = [290, 195, 0]; landmarks[38] = [295, 195, 0]
        landmarks[40] = [295, 205, 0]; landmarks[41] = [290, 205, 0]
        
        landmarks[42] = [340, 200, 0]; landmarks[45] = [360, 200, 0]
        landmarks[43] = [345, 195, 0]; landmarks[44] = [355, 195, 0]
        landmarks[46] = [355, 205, 0]; landmarks[47] = [350, 205, 0]
        
        # Mouth (48, 54)
        landmarks[48] = [290, 280, 0]
        landmarks[54] = [350, 280, 0]

        # V4 now uses get_landmarks_from_image for 3D
        mock_fa.get_landmarks_from_image.return_value = [landmarks]

        # Mock SixDRepNet
        mock_pose_model = MagicMock()
        mock_SixDRepNet.return_value = mock_pose_model
        # predict returns pitch, yaw, roll (batch size 1)
        mock_pose_model.predict.return_value = (np.array([10.0]), np.array([20.0]), np.array([5.0]))

        state = {
            "data_dir": self.test_dir,
            "debug": True
        }

        new_state = v4_blink_headpose_dynamics.run(state)

        self.assertIn("blink_data", new_state)
        self.assertIn("head_pose_data", new_state)
        self.assertIn("headpose_viz_path", new_state)
        
        blink_data = new_state["blink_data"]
        self.assertEqual(len(blink_data), 2) # 2 frames processed
        self.assertTrue(blink_data[0]["ear"] > 0)
        
        pose_data = new_state["head_pose_data"]
        self.assertEqual(len(pose_data), 2)
        self.assertIn("pitch", pose_data[0]["pose"])
        self.assertIn("yaw", pose_data[0]["pose"])
        self.assertIn("roll", pose_data[0]["pose"])

        print("V4 extraction test passed.")

    def test_missing_video(self):
        print("\nTesting missing video for V4...")
        state = {"data_dir": "non_existent_dir"}
        new_state = v4_blink_headpose_dynamics.run(state)
        self.assertEqual(new_state, state)
        print("V4 missing video test passed.")

if __name__ == "__main__":
    unittest.main()
