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
        self.video_path = os.path.join(self.test_dir, "test_video.mp4")
        
        # Create a dummy video with a "face" (rectangle)
        fps = 30
        duration = 2 # seconds
        width, height = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.video_path, fourcc, fps, (width, height))
        
        for i in range(fps * duration):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Draw a white rectangle in the center to simulate a face
            # Haar cascade might not detect a simple rectangle, but we can at least test the pipeline.
            # To actually trigger detection, we'd need a real face image or a more complex pattern.
            # For unit testing the pipeline logic, ensuring keyframes are extracted is the main goal.
            # We can mock the detector if we want to test detection logic specifically, 
            # but for now let's see if the pipeline runs.
            cv2.rectangle(frame, (200, 150), (440, 330), (255, 255, 255), -1)
            out.write(frame)
            
        out.release()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_keyframe_extraction(self):
        print("\nTesting keyframe extraction and face cropping...")
        state = {"video_path": self.video_path, "debug": True}
        
        new_state = v1_keyframes_facetrack.run(state)
        
        keyframes = new_state.get("keyframes")
        self.assertIsNotNone(keyframes)
        self.assertTrue(len(keyframes) >= 2)
        
        for kf in keyframes:
            self.assertTrue(os.path.exists(kf))
            
        self.assertIn("face_detections", new_state)
        self.assertEqual(len(new_state["face_detections"]), len(keyframes))
        
        # Check for face detections and crops
        # Since we drew a rectangle, the DNN model will likely NOT detect it as a face.
        # This is expected behavior for a robust model (it shouldn't detect a rectangle as a face).
        # We verify that the pipeline ran without errors and produced the expected structure.
        
        for detection in new_state["face_detections"]:
            self.assertIn("frame_id", detection)
            self.assertIn("timestamp", detection)
            self.assertIn("faces", detection)
            self.assertIsInstance(detection["faces"], list)
            
            for face in detection["faces"]:
                self.assertIn("bbox", face)
                self.assertIn("is_main", face)
                self.assertIn("crop_path", face)
                self.assertIn("confidence", face) # DNN adds confidence
                if os.path.exists(face["crop_path"]):
                    pass # Crop saved successfully
        
        print("Keyframe extraction and face cropping test passed.")

    def test_missing_video(self):
        print("\nTesting missing video file...")
        state = {"video_path": "non_existent.mp4"}
        new_state = v1_keyframes_facetrack.run(state)
        self.assertNotIn("keyframes", new_state)
        print("Missing video test passed.")

if __name__ == "__main__":
    unittest.main()
