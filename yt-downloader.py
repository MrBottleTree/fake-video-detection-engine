import yt_dlp
import sys

import shutil
try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_PATH = None

def download_video(url):
    """
    Downloads the highest quality video from the given YouTube URL as MP4.
    """
    # Check if system ffmpeg is available, otherwise use imageio-ffmpeg
    system_ffmpeg = shutil.which('ffmpeg')
    ffmpeg_location = system_ffmpeg if system_ffmpeg else FFMPEG_PATH
    
    if ffmpeg_location:
        print(f"FFmpeg found at: {ffmpeg_location}")
        print("Downloading highest quality H.264 video (most compatible)...")
        ydl_opts = {
            # Prefer AVC (H.264) video and AAC audio for maximum compatibility
            'format': 'bestvideo[vcodec^=avc]+bestaudio[acodec^=mp4a]/best[ext=mp4]/best',
            'merge_output_format': 'mp4',
            'outtmpl': 'videos/%(title)s.%(ext)s',
            'ffmpeg_location': ffmpeg_location,
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
            'noplaylist': True,
        }
    else:
        print("WARNING: FFmpeg NOT found. Falling back to best single file.")
        print("High quality (1080p+) requires FFmpeg. Please install 'imageio-ffmpeg' or system ffmpeg.")
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': 'videos/%(title)s.%(ext)s',
            'noplaylist': True,
        }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading video from: {url}")
            ydl.download([url])
            print("\nDownload completed successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_url = sys.argv[1]
    else:
        video_url = input("Enter the YouTube video URL: ")
    
    if video_url:
        download_video(video_url)
    else:
        print("No URL provided.")
