import argparse
from langgraph.graph import StateGraph, END
from nodes import *
import os
import sys
import shutil
import datetime
import yt_dlp
from moviepy import VideoFileClip
import imageio_ffmpeg
from typing import Optional, Dict, Any, TypedDict, Annotated
import operator
import warnings
warnings.filterwarnings("ignore", message=".*'pin_memory' argument is set as true.*")

def overwrite(left, right):
    return right

class State(TypedDict):
    input_path: Annotated[str, overwrite]
    label: Annotated[Optional[int], overwrite]
    data_dir: Annotated[Optional[str], overwrite]
    metadata: Annotated[Optional[Dict[str, Any]], overwrite]
    fake_probability: Annotated[Optional[float], overwrite]
    debug: Annotated[bool, overwrite]
    
    transcript: Annotated[Optional[str], overwrite]
    segments: Annotated[Optional[list], overwrite]
    word_count: Annotated[Optional[int], overwrite]
    audio_onsets: Annotated[Optional[list], overwrite]
    onset_count: Annotated[Optional[int], overwrite]
    
    keyframes: Annotated[Optional[list], overwrite]
    face_detections: Annotated[Optional[list], overwrite]
    ocr_results: Annotated[Optional[list], overwrite]
    
    mouth_landmarks: Annotated[Optional[list], overwrite]
    mouth_landmarks_viz_path: Annotated[Optional[str], overwrite]
    
    blink_data: Annotated[Optional[list], overwrite]
    head_pose_data: Annotated[Optional[list], overwrite]
    headpose_viz_path: Annotated[Optional[str], overwrite]
    
    lip_sync_score: Annotated[Optional[float], overwrite]
    claims: Annotated[Optional[list], overwrite]
    evidence: Annotated[Optional[list], overwrite]
    features: Annotated[Optional[Dict[str, float]], overwrite]

def in_node(state: State) -> State:
    input_path = state["input_path"]
    debug = state.get("debug", False)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("processed", f"video_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = {}

    if input_path.startswith("http://") or input_path.startswith("https://"):
        print(f"Downloading video from URL: {input_path}")
        
        max_retries = 3
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                cookies_path = "cookies.txt"
                use_cookies_txt = os.path.exists(cookies_path)
                
                ydl_opts = {
                    'outtmpl': os.path.join(output_dir, 'video.%(ext)s'),
                    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                    'ffmpeg_location': imageio_ffmpeg.get_ffmpeg_exe(),
                    'quiet': not debug,
                    'no_warnings': not debug,
                    'sleep_interval': 2,
                    'max_sleep_interval': 5,
                    'sleep_interval_requests': 1,
                    'extractor_args': {
                        'youtube': {
                            'player_client': ['default', 'ios', 'android', 'web'],
                            'player_skip': ['webpage', 'configs', 'js'],
                        }
                    },
                    'retry_sleep_functions': {'http': lambda n: 5},
                }
                
                if use_cookies_txt:
                    print(f"Using cookies from {cookies_path}")
                    ydl_opts['cookiefile'] = cookies_path
                else:
                    print("No cookies.txt found. Using browser simulation (extractor_args).")

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(input_path, download=True)
                    video_path = ydl.prepare_filename(info)
                    metadata = {
                        "title": info.get("title"),
                        "duration": info.get("duration"),
                        "uploader": info.get("uploader"),
                        "original_url": input_path
                    }
                    success = True
                    
                    # Save a copy to 'videos' folder as requested
                    try:
                        videos_dir = "videos"
                        os.makedirs(videos_dir, exist_ok=True)
                        saved_video_path = os.path.join(videos_dir, f"video_{timestamp}.mp4")
                        shutil.copy2(video_path, saved_video_path)
                        print(f"Video saved to: {saved_video_path}")
                    except Exception as save_err:
                        print(f"Warning: Failed to save copy to 'videos' folder: {save_err}")
                    
            except yt_dlp.utils.DownloadError as e:
                retry_count += 1
                print(f"Download attempt {retry_count} failed: {e}")
                
                if retry_count < max_retries:
                    import time
                    wait_time = retry_count * 5
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    
                    if retry_count == max_retries - 1:
                        print("\n[ESCALATION] Standard download failed. Attempting OAuth2 authentication...")
                        print("Please watch the console for a Google Device Code.")
                        print("You will need to visit google.com/device and enter the code.")
                        ydl_opts['username'] = 'oauth2'
                        ydl_opts['password'] = ''
                else:
                    print("All download attempts failed.")
                    print("CRITICAL TIP FOR EC2: Upload a 'cookies.txt' file to the project root.")
                    print("Alternatively, use the OAuth2 flow if prompted above.")
                    raise e
    else:
        print(f"Processing local file: {input_path}")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        video_path = os.path.join(output_dir, "video.mp4")
        shutil.copy2(input_path, video_path)
        metadata = {"original_path": input_path}


    print(f"Extracting audio from: {video_path}")
    try:
        clip = VideoFileClip(video_path)
        audio_path = os.path.join(output_dir, "audio.wav")
        
        if clip.audio:
            clip.audio.write_audiofile(audio_path, logger=None)
        else:
            print("Warning: No audio track found in video.")
            audio_path = None

        metadata.update({
            "duration": clip.duration,
            "fps": clip.fps,
            "size": clip.size,
            "rotation": getattr(clip, "rotation", 0)
        })
        
        clip.close()
    except Exception as e:
        print(f"Error processing video with MoviePy: {e}")
        raise e

    print(f"Processing complete. Video: {video_path}, Audio: {audio_path}")

    state["data_dir"] = output_dir
    state["metadata"] = metadata
    state["debug"] = debug
    return state

graph = StateGraph(State)

graph.add_node("IN", in_node)

graph.add_node("A1", a1_demux_audio_extract.run)
graph.add_node("A2", a2_vad_asr.run)
graph.add_node("A3", a3_audio_onsets.run)

graph.add_node("V1", v1_keyframes_facetrack.run)
graph.add_node("V2", v2_ocr_overlays.run)
graph.add_node("V3", v3_mouth_landmarks_timeseries.run)
graph.add_node("V4", v4_blink_headpose_dynamics.run)
graph.add_node("V5", v5_texture_ela.run)

graph.add_node("C1", c1_lip_sync_score.run)
graph.add_node("C2", c2_gesture_narration_check.run)
graph.add_node("C3", c3_claim_extraction.run)

graph.add_node("E1", e1_web_evidence.run)
graph.add_node("E2", e2_source_reliability.run)
graph.add_node("E3", e3_claim_evidence_scorer.run)

graph.add_node("LR", lr_node.run)

graph.set_entry_point("IN")
graph.add_edge("IN", "A1")
graph.add_edge("IN", "V1")
graph.add_edge("A1", "A2")
graph.add_edge("A1", "A3")
graph.add_edge("V1", "V2")
graph.add_edge("V1", "V3")
graph.add_edge("V1", "V4")
graph.add_edge("V1", "V5")
graph.add_edge("A3", "C1")
graph.add_edge("V3", "C1")
graph.add_edge("A2", "C2")
graph.add_edge("V1", "C2")
graph.add_edge("A2", "C3")
graph.add_edge("V2", "C3")
graph.add_edge("C3", "E1")
graph.add_edge("E1", "E2")
graph.add_edge("E2", "E3")
graph.add_edge("C3", "E3")
graph.add_edge("E1", "E3")
graph.add_edge("C1", "LR")
graph.add_edge("C2", "LR")
graph.add_edge("V4", "LR")
graph.add_edge("V5", "LR")
graph.add_edge("E3", "LR")

graph.add_edge("LR", END)

app = graph.compile()

def main() -> None:
    parser = argparse.ArgumentParser(description="Fake Video Detection Engine")
    parser.add_argument("input_path", help="Path to video file or URL")
    parser.add_argument("label", nargs="?", type=int, help="Optional label (0 or 1)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()

    if args.label is not None and args.label not in [0, 1]:
        print("Label must be 0 or 1 if provided.")
        sys.exit(1)

    state = {
        "input_path": args.input_path,
        "debug": args.debug
    }
    
    if args.label is not None:
        state["label"] = args.label

    print(f"Starting processing with debug={'ON' if args.debug else 'OFF'}...")
    result = app.invoke(state)

    fake_prob = result.get("fake_probability")
    if args.debug:
        print("Final state:", result)
    print("Fake probability:", fake_prob)


if __name__ == "__main__":
    main()
