import unittest
import os
import shutil
import sys
import wave
import contextlib
from unittest.mock import MagicMock, patch
import numpy as np
from scipy.io import wavfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.A_nodes import a1_demux_audio_extract

class TestA1AudioStandardization(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_audio_processing"
        os.makedirs(self.test_dir, exist_ok=True)
        self.input_audio = os.path.join(self.test_dir, "audio.wav")
        
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        stereo_data = np.vstack((audio_data, audio_data)).T
        wavfile.write(self.input_audio, sample_rate, stereo_data.astype(np.float32))

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_audio_standardization(self):
        print("\nTesting audio standardization...")
        state = {"data_dir": self.test_dir}
        
        new_state = a1_demux_audio_extract.run(state)
        
        output_path = os.path.join(self.test_dir, "audio_16k.wav")
        self.assertTrue(os.path.exists(output_path))
        
        with contextlib.closing(wave.open(output_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            channels = f.getnchannels()
            duration = frames / float(rate)
            
            self.assertEqual(rate, 16000)
            self.assertEqual(channels, 1)
            self.assertAlmostEqual(duration, 1.0, delta=0.1)

        self.assertEqual(new_state["metadata"]["audio_sample_rate"], 16000)
        self.assertEqual(new_state["metadata"]["audio_channels"], 1)
        print("Audio standardization test passed.")

    def test_missing_data_dir(self):
        print("\nTesting missing data directory...")
        state = {"data_dir": "non_existent_dir"}
        new_state = a1_demux_audio_extract.run(state)
        self.assertEqual(new_state["data_dir"], "non_existent_dir")
        print("Missing data directory test passed.")

if __name__ == "__main__":
    unittest.main()
