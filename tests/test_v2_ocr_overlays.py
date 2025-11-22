import unittest
import os
import sys
import cv2
import numpy as np
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.V_nodes import v2_ocr_overlays

class TestV2OCROverlays(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_v2_ocr"
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.keyframe_path = os.path.join(self.test_dir, "test_text.jpg")
        
        img = np.ones((200, 600, 3), dtype=np.uint8) * 255
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'HELLO WORLD', (50, 100), font, 2, (0, 0, 0), 3, cv2.LINE_AA)
        
        cv2.imwrite(self.keyframe_path, img)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_ocr_extraction(self):
        print("\nTesting OCR extraction...")
        state = {
            "keyframes": [self.keyframe_path],
            "debug": True
        }
        
        new_state = v2_ocr_overlays.run(state)
        
        ocr_results = new_state.get("ocr_results")
        self.assertIsNotNone(ocr_results)
        self.assertTrue(len(ocr_results) > 0)
        
        detections = ocr_results[0]["detections"]
        self.assertTrue(len(detections) > 0)
        
        detected_text = " ".join([d["text"] for d in detections])
        print(f"Detected text: {detected_text}")
        
        self.assertIn("HELLO", detected_text.upper())
        self.assertIn("WORLD", detected_text.upper())
        
        print("OCR extraction test passed.")

    def test_no_keyframes(self):
        print("\nTesting no keyframes...")
        state = {"keyframes": [], "debug": True}
        new_state = v2_ocr_overlays.run(state)
        self.assertEqual(new_state, state)
        print("No keyframes test passed.")

if __name__ == "__main__":
    unittest.main()
