import unittest
import os
import shutil
import sys
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import in_node, State
from moviepy import ColorClip

class TestInNode(unittest.TestCase):
    def setUp(self):
        self.test_video = "test_video.mp4"
        clip = ColorClip(size=(640, 480), color=(255, 0, 0), duration=2)
        clip.fps = 24
        clip.write_videofile(self.test_video, codec="libx264", audio=False, logger=None)
        self.created_dirs = []

    def tearDown(self):
        if os.path.exists(self.test_video):
            os.remove(self.test_video)
        for d in self.created_dirs:
            if os.path.exists(d):
                shutil.rmtree(d)

    def test_local_file_ingestion(self):
        print("\nTesting local file ingestion...")
        state = State(input_path=os.path.abspath(self.test_video))
        
        new_state = in_node(state)
        
        if new_state.get("video_path"):
            parent_dir = os.path.dirname(new_state["video_path"])
            if parent_dir not in self.created_dirs:
                self.created_dirs.append(parent_dir)

        self.assertIsNotNone(new_state.get("video_path"))
        self.assertTrue(os.path.exists(new_state["video_path"]))
        self.assertIsNone(new_state.get("audio_path"))
        
        metadata = new_state.get("metadata")
        self.assertIsNotNone(metadata)
        self.assertAlmostEqual(metadata["duration"], 2, delta=0.1)
        
        print("Local file ingestion test passed.")

    def test_invalid_local_file(self):
        print("\nTesting invalid local file...")
        state = State(input_path="non_existent_video.mp4")
        with self.assertRaises(FileNotFoundError):
            in_node(state)
        print("Invalid local file test passed.")

    @patch('main.yt_dlp.YoutubeDL')
    @patch('main.VideoFileClip')
    def test_url_ingestion(self, mock_video_clip, mock_ydl):
        print("\nTesting URL ingestion (Mocked)...")
        
        mock_ydl_instance = MagicMock()
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance
        
        mock_info = {
            "title": "Test Video",
            "duration": 10,
            "uploader": "Test Uploader",
            "ext": "mp4"
        }
        mock_ydl_instance.extract_info.return_value = mock_info
        mock_ydl_instance.prepare_filename.return_value = "processed/mock_video/video.mp4"
        
        mock_clip_instance = MagicMock()
        mock_video_clip.return_value = mock_clip_instance
        mock_clip_instance.duration = 10
        mock_clip_instance.fps = 30
        mock_clip_instance.size = [1920, 1080]
        mock_clip_instance.rotation = 0
        mock_clip_instance.audio = MagicMock()
        
        state = State(input_path="https://www.youtube.com/watch?v=test")

        new_state = in_node(state)
        
        mock_ydl.assert_called()
        mock_ydl_instance.extract_info.assert_called_with("https://www.youtube.com/watch?v=test", download=True)
        
        self.assertEqual(new_state["video_path"], "processed/mock_video/video.mp4")
        self.assertIsNotNone(new_state["audio_path"])
        self.assertEqual(new_state["metadata"]["title"], "Test Video")
        
        print("URL ingestion test passed.")

if __name__ == "__main__":
    unittest.main()
