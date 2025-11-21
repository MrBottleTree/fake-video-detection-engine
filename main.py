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

def overwrite(left, right):
    return right

class State(TypedDict):
    input_path: Annotated[str, overwrite]
    label: Annotated[Optional[int], overwrite]
    video_path: Annotated[Optional[str], overwrite]
    audio_path: Annotated[Optional[str], overwrite]
    metadata: Annotated[Optional[Dict[str, Any]], overwrite]
    fake_probability: Annotated[Optional[float], overwrite]

def in_node(state: State) -> State:
    input_path = state["input_path"]
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("processed", f"video_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    video_path = ""
    metadata = {}

    if input_path.startswith("http://") or input_path.startswith("https://"):
        print(f"Downloading video from URL: {input_path}")
        ydl_opts = {
            'outtmpl': os.path.join(output_dir, 'video.%(ext)s'),
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'ffmpeg_location': imageio_ffmpeg.get_ffmpeg_exe(),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(input_path, download=True)
            video_path = ydl.prepare_filename(info)
            metadata = {
                "title": info.get("title"),
                "duration": info.get("duration"),
                "uploader": info.get("uploader"),
                "original_url": input_path
            }
    else:
        print(f"Processing local file: {input_path}")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        filename = os.path.basename(input_path)
        video_path = os.path.join(output_dir, filename)
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
    
    print(f"Processing complete. Video: {video_path}, Audio: {audio_path}")
    
    return {
        "video_path": video_path,
        "audio_path": audio_path,
        "metadata": metadata
    }

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
graph.add_edge("C1", "LR")
graph.add_edge("C2", "LR")
graph.add_edge("V4", "LR")
graph.add_edge("V5", "LR")
graph.add_edge("E3", "LR")

graph.add_edge("LR", END)

app = graph.compile()

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_or_url> [label_0_or_1]")
        sys.exit(1)

    input_path = sys.argv[1]
    state = {"input_path": input_path}

    if len(sys.argv) >= 3:
        try:
            state["label"] = int(sys.argv[2])
        except ValueError:
            print("Label must be 0 or 1 if provided.")
            sys.exit(1)

    result = app.invoke(state)

    fake_prob = result.get("fake_probability")
    print("Final state:", result)
    print("Fake probability:", fake_prob)


if __name__ == "__main__":
    main()
