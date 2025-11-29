import os
import librosa
import numpy as np
from nodes import dump_node_debug
import sys

print("Module A3 imported", flush=True)

def run(state: dict) -> dict:
    print("Node A3: Detecting audio onsets and envelope...", flush=True)
    audio_path = os.path.join(state.get("data_dir"), "audio_16k.wav")
    debug = state.get("debug", False)
    
    if not audio_path or not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return state

    try:
        if debug:
            print(f"Loading audio from {audio_path}...")
        y, sr = librosa.load(audio_path, sr=None)
        
        if debug:
            print("Detecting onsets...")
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        onset_times_list = onset_times.tolist()
        
        print(f"Detected {len(onset_times_list)} onsets.")
        
        state["audio_onsets"] = onset_times_list
        state["onset_count"] = len(onset_times_list)
        
        # --- New: Calculate Audio Envelope (RMS) ---
        # We target a specific FPS if available, otherwise default to 30fps for envelope
        metadata = state.get("metadata", {})
        fps = metadata.get("fps", 30.0)
        duration = metadata.get("duration")
        
        hop_length = int(sr / fps)
        rms = librosa.feature.rms(y=y, frame_length=hop_length*2, hop_length=hop_length, center=True)[0]
        
        # Interpolate to match exact number of frames if duration is known
        if duration:
            target_frames = int(duration * fps)
            if len(rms) != target_frames:
                rms = np.interp(
                    np.linspace(0, 1, target_frames),
                    np.linspace(0, 1, len(rms)),
                    rms
                )
        
        state["audio_envelope"] = rms.tolist()
        # -------------------------------------------
        
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"]["onset_detection_method"] = "librosa.onset.onset_detect"
        dump_node_debug(
            state,
            "A3",
            {
                "onset_count": len(onset_times_list),
                "envelope_len": len(state.get("audio_envelope", [])),
                "fps": fps,
            },
        )

    except Exception as e:
        print(f"Error in A3 node: {e}")
        raise e

    if state.get("debug", False):
        print(f"[DEBUG] A3: Onset Detection Method: librosa.onset.onset_detect")
        print(f"[DEBUG] A3: Total Onsets: {state.get('onset_count')}")
        onsets = state.get("audio_onsets", [])
        if onsets:
            print(f"[DEBUG] A3: First 5 Onsets: {onsets[:5]}")
        print(f"[DEBUG] A3: Audio Envelope Length: {len(state.get('audio_envelope', []))}")

    return state
