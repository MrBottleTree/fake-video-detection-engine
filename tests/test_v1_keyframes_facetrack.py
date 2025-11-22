import unittest
import os
import sys
import cv2
import numpy as np
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.V_nodes import v1_keyframes_facetrack

class TestV1KeyframesFacetrack(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_v1_keyframes"
        os.makedirs(self.test_dir, exist_ok=True)
        self.video_path = os.path.join(self.test_dir, "video.mp4")
        
        fps = 30
        duration = 2
        width, height = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.video_path, fourcc, fps, (width, height))
        
        for i in range(fps * duration):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.rectangle(frame, (200, 150), (440, 330), (255, 255, 255), -1)
            out.write(frame)
            
        out.release()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_keyframe_extraction(self):
        print("\nTesting keyframe extraction and face cropping...")
        state = {"data_dir": self.test_dir, "debug": True}
        
        new_state = v1_keyframes_facetrack.run(state)
        
        keyframes = new_state.get("keyframes")
        self.assertIsNotNone(keyframes)
        self.assertTrue(len(keyframes) >= 2)
        
        for kf in keyframes:
            self.assertTrue(os.path.exists(kf))
            
        self.assertIn("face_detections", new_state)
        self.assertEqual(len(new_state["face_detections"]), len(keyframes))
                
        for detection in new_state["face_detections"]:
            self.assertIn("frame_id", detection)
            self.assertIn("timestamp", detection)
            self.assertIn("faces", detection)
            self.assertIsInstance(detection["faces"], list)
            
            for face in detection["faces"]:
                self.assertIn("bbox", face)
                self.assertIn("is_main", face)
                self.assertIn("crop_path", face)
                self.assertIn("confidence", face)
                if os.path.exists(face["crop_path"]):
                    pass
        
        print("Keyframe extraction and face cropping test passed.")

    def test_missing_video(self):
        print("\nTesting missing video file...")
        state = {"data_dir": "non_existent_dir"}
        new_state = v1_keyframes_facetrack.run(state)
        self.assertNotIn("keyframes", new_state)
        print("Missing video test passed.")

if __name__ == "__main__":
    unittest.main()
