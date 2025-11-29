import os
import subprocess
import shutil
import sys

def debug_reencode():
    # Find the latest video directory
    processed_dir = "processed"
    if not os.path.exists(processed_dir):
        print("No processed directory found.")
        return

    subdirs = [os.path.join(processed_dir, d) for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    if not subdirs:
        print("No subdirectories in processed.")
        return
        
    latest_dir = max(subdirs, key=os.path.getmtime)
    video_path = os.path.join(latest_dir, "video.mp4")
    print(f"Checking video: {video_path}")
    
    if not os.path.exists(video_path):
        print("video.mp4 not found.")
        return

    # Check Codec
    probe_cmd = [
        "ffprobe", 
        "-v", "error", 
        "-select_streams", "v:0", 
        "-show_entries", "stream=codec_name", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        video_path
    ]
    try:
        codec = subprocess.check_output(probe_cmd).decode("utf-8").strip()
        print(f"Raw Codec Output: '{codec}'")
    except Exception as e:
        print(f"ffprobe failed: {e}")
        codec = "unknown"

    # Check FPS
    probe_fps_cmd = [
        "ffprobe", 
        "-v", "error", 
        "-select_streams", "v:0", 
        "-show_entries", "stream=r_frame_rate", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        video_path
    ]
    try:
        fps_str = subprocess.check_output(probe_fps_cmd).decode("utf-8").strip()
        print(f"Raw FPS Output: '{fps_str}'")
        num, den = map(int, fps_str.split('/'))
        current_fps = num / den
        print(f"Parsed FPS: {current_fps}")
    except Exception as e:
        print(f"FPS check failed: {e}")
        current_fps = 30.0

    if codec != "h264" or current_fps > 30.5:
        print("Condition MET: Re-encoding required.")
        
        ffmpeg_exe = shutil.which("ffmpeg")
        print(f"System ffmpeg: {ffmpeg_exe}")
        
        temp_path = os.path.join(latest_dir, "debug_temp.mp4")
        cmd = [
            ffmpeg_exe, "-y",
            "-i", video_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-r", "30",
            "-c:a", "aac",
            "-strict", "experimental",
            temp_path
        ]
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
            print("ffmpeg command finished.")
            
            if os.path.exists(temp_path):
                size = os.path.getsize(temp_path)
                print(f"Temp file size: {size}")
                if size > 0:
                    print("Replacing original file...")
                    os.replace(temp_path, video_path)
                    print("Done.")
                else:
                    print("Temp file is empty.")
            else:
                print("Temp file not created.")
                
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed with code {e.returncode}")
    else:
        print("Condition NOT met. Skipping re-encode.")

if __name__ == "__main__":
    debug_reencode()
