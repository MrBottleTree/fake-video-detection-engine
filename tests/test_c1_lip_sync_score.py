import unittest
import os
import sys
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from nodes.C_nodes.c1_lip_sync_score import run as c1_run

class TestC1LipSyncScore(unittest.TestCase):

    def setUp(self):
        self.base_state: State = {
            "metadata": {"duration": 5.0, "fps": 30.0},
            "debug": False,
            "input_path": "", "label": None, "data_dir": None,
            "fake_probability": None, "transcript": None, "segments": None,
            "word_count": None, "onset_count": None, "keyframes": None,
            "face_detections": None, "ocr_results": None, "mouth_landmarks_viz_path": None,
            "blink_data": None, "head_pose_data": None, "headpose_viz_path": None,
            "audio_onsets": [], "mouth_landmarks": [], "lip_sync_score": None
        }

    @staticmethod
    def generate_mouth_data(timestamps, mar_peaks_times, base_mar=0.05, peak_mar=0.7, peak_width=0.1):
        landmarks = []
        for t in timestamps:
            mar = base_mar
            if any(abs(t - peak_time) < peak_width for peak_time in mar_peaks_times):
                mar = peak_mar
            landmarks.append({"timestamp": t, "mar": mar})
        return landmarks

    def test_perfect_sync(self):
        """
        Tests a scenario where mouth MAR peaks exactly at audio onsets.
        The expected score should be very high.
        """
        state = self.base_state.copy()
        duration = state["metadata"]["duration"]
        fps = state["metadata"]["fps"]
        num_frames = int(duration * fps)
        timestamps = np.linspace(0, duration, num_frames)
        
        onset_times = [1.0, 2.5, 4.0]
        state["audio_onsets"] = onset_times
        state["mouth_landmarks"] = self.generate_mouth_data(timestamps, mar_peaks_times=onset_times)
        
        result_state = c1_run(state)
        
        self.assertIn("lip_sync_score", result_state)
        self.assertIsNotNone(result_state["lip_sync_score"])
        self.assertGreater(result_state["lip_sync_score"], 0.8, "Score should be high for perfect sync")

    def test_delayed_sync(self):
        """
        Tests a scenario where mouth movements are slightly delayed.
        Cross-correlation should still find a high score.
        """
        state = self.base_state.copy()
        duration = state["metadata"]["duration"]
        fps = state["metadata"]["fps"]
        num_frames = int(duration * fps)
        timestamps = np.linspace(0, duration, num_frames)
        
        onset_times = [1.0, 2.5, 4.0]
        delay = 0.12  # 120ms delay
        mouth_peak_times = [t + delay for t in onset_times]
        state["audio_onsets"] = onset_times
        state["mouth_landmarks"] = self.generate_mouth_data(timestamps, mar_peaks_times=mouth_peak_times)
        
        result_state = c1_run(state)
        
        self.assertIn("lip_sync_score", result_state)
        self.assertGreater(result_state["lip_sync_score"], 0.7, "Score should be high for delayed sync")

    def test_no_sync(self):
        """
        Tests where mouth movements are unrelated to audio onsets.
        The expected score should be very low.
        """
        state = self.base_state.copy()
        duration = state["metadata"]["duration"]
        fps = state["metadata"]["fps"]
        num_frames = int(duration * fps)
        timestamps = np.linspace(0, duration, num_frames)
        
        onset_times = [1.0, 2.5, 4.0]
        mouth_peak_times = [0.5, 1.8, 3.3] # Unrelated times
        state["audio_onsets"] = onset_times
        state["mouth_landmarks"] = self.generate_mouth_data(timestamps, mar_peaks_times=mouth_peak_times)
        
        result_state = c1_run(state)
        
        self.assertIn("lip_sync_score", result_state)
        self.assertLess(result_state["lip_sync_score"], 0.25, "Score should be low for no sync")

    def test_anti_sync(self):
        """
        Tests where the mouth is closed during audio onsets.
        The score should be very low, not necessarily exactly zero.
        """
        state = self.base_state.copy()
        duration = state["metadata"]["duration"]
        fps = state["metadata"]["fps"]
        num_frames = int(duration * fps)
        timestamps = np.linspace(0, duration, num_frames)
        onset_times = [1.0, 2.5, 4.0]

        landmarks = [{"timestamp": t, "mar": 0.05} for t in timestamps]
        for lm in landmarks:
            if not any(abs(lm['timestamp'] - onset) < 0.1 for onset in onset_times):
                lm['mar'] = 0.7
        
        state["audio_onsets"] = onset_times
        state["mouth_landmarks"] = landmarks

        result_state = c1_run(state)

        self.assertIn("lip_sync_score", result_state)
        self.assertLess(result_state["lip_sync_score"], 0.25, "Score should be very low for anti-sync")
    
    def test_missing_data_returns_zero(self):
        """Tests that missing essential data lists results in a score of 0.0."""
        with self.subTest(msg="Missing mouth_landmarks"):
            state = self.base_state.copy()
            state["audio_onsets"] = [1.0]
            state["mouth_landmarks"] = [] 
            result_state = c1_run(state)
            self.assertEqual(result_state["lip_sync_score"], 0.0)

        with self.subTest(msg="Missing audio_onsets"):
            state = self.base_state.copy()
            state["audio_onsets"] = [] 
            state["mouth_landmarks"] = [{"timestamp": 1.0, "mar": 0.5}]
            result_state = c1_run(state)
            self.assertEqual(result_state["lip_sync_score"], 0.0)

    def test_missing_metadata_returns_zero(self):
        """Tests that missing essential metadata results in a score of 0.0."""
        with self.subTest(msg="Missing metadata 'fps'"):
            state = self.base_state.copy()
            state["audio_onsets"] = [1.0]
            state["mouth_landmarks"] = [{"timestamp": 1.0, "mar": 0.5}]
            del state["metadata"]["fps"]
            result_state = c1_run(state)
            self.assertEqual(result_state["lip_sync_score"], 0.0)

        with self.subTest(msg="Missing metadata 'duration'"):
            state = self.base_state.copy()
            state["audio_onsets"] = [1.0]
            state["mouth_landmarks"] = [{"timestamp": 1.0, "mar": 0.5}]
            del state["metadata"]["duration"]
            result_state = c1_run(state)
            self.assertEqual(result_state["lip_sync_score"], 0.0)

    def test_no_activity_returns_zero(self):
        """Tests that no variation in signals results in a score of 0.0."""
        duration = self.base_state["metadata"]["duration"]
        fps = self.base_state["metadata"]["fps"]
        num_frames = int(duration * fps)
        timestamps = np.linspace(0, duration, num_frames)

        with self.subTest(msg="No audio activity"):
            state = self.base_state.copy()
            state["audio_onsets"] = []
            state["mouth_landmarks"] = self.generate_mouth_data(timestamps, mar_peaks_times=[1.0, 2.5])
            result_state = c1_run(state)
            self.assertEqual(result_state["lip_sync_score"], 0.0)
        
        with self.subTest(msg="No mouth activity"):
            state = self.base_state.copy()
            state["audio_onsets"] = [1.0, 2.5]
            state["mouth_landmarks"] = [{"timestamp": t, "mar": 0.1} for t in timestamps]
            result_state = c1_run(state)
            self.assertAlmostEqual(result_state["lip_sync_score"], 0.0, places=7)


if __name__ == "__main__":
    unittest.main()