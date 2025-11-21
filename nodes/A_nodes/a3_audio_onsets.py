import os
import librosa
import numpy as np

def run(state: dict) -> dict:
    print("Node A3: Detecting audio onsets...")
    audio_path = state.get("audio_path")
    
    if not audio_path or not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return state

    try:
        print(f"Loading audio from {audio_path}...")
        y, sr = librosa.load(audio_path, sr=None)
        
        print("Detecting onsets...")
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        onset_times_list = onset_times.tolist()
        
        print(f"Detected {len(onset_times_list)} onsets.")
        
        state["audio_onsets"] = onset_times_list
        state["onset_count"] = len(onset_times_list)
        
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"]["onset_detection_method"] = "librosa.onset.onset_detect"

    except Exception as e:
        print(f"Error in A3 node: {e}")
        raise e

    if state.get("debug", False):
        print(f"[DEBUG] A3: Onset Detection Method: librosa.onset.onset_detect")
        print(f"[DEBUG] A3: Total Onsets: {state.get('onset_count')}")
        onsets = state.get("audio_onsets", [])
        if onsets:
            print(f"[DEBUG] A3: First 5 Onsets: {onsets[:5]}")

    return state
