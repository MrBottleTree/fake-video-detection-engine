import numpy as np
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import distance as dist
from typing import TypedDict, Optional, Dict, Any, Annotated
import operator

def calculate_mar(mouth_points):
    if len(mouth_points) < 20:
        return 0.0
        
    m = mouth_points
    
    # Vertical distances
    A = dist.euclidean(m[2], m[10])
    B = dist.euclidean(m[3], m[9])
    C = dist.euclidean(m[4], m[8])
    
    # Horizontal distance
    D = dist.euclidean(m[0], m[6])
    
    if D == 0:
        return 0.0
        
    mar = (A + B + C) / (3.0 * D)
    return mar

def run(state: dict) -> dict:
    print("Node C1: Analyzing Lip Sync (Robust MAR Calculation)...")

    mouth_landmarks = state.get("mouth_landmarks")
    audio_onsets = state.get("audio_onsets")
    metadata = state.get("metadata", {})
    fps = metadata.get("fps")
    duration = metadata.get("duration")

    if not mouth_landmarks or not audio_onsets:
        print(" C1: Warning - Missing mouth landmarks or audio onsets. Cannot compute lip-sync score.")
        state["lip_sync_score"] = 0.0 #default bad score
        return state

    if not fps or not duration:
        print(" C1: Warning - Missing video FPS or duration from metadata. Cannot compute lip-sync score.")
        state["lip_sync_score"] = 0.0
        return state

    
    #both signals should be compared with same time axis
    num_frames = int(duration * fps)
    time_axis = np.linspace(0, duration, num_frames)

    # Calculate MAR for each frame
    mouth_timestamps = []
    mouth_mar_values = []
    
    for lm in mouth_landmarks:
        timestamp = lm.get('timestamp', 0.0)
        
        if 'mar' in lm:
            mar = lm['mar']
        elif 'landmarks' in lm:
            points = lm['landmarks']
            if not points:
                mar = 0.0
            else:
                mar = calculate_mar(points)
        else:
            mar = 0.0
            
        mouth_timestamps.append(timestamp)
        mouth_mar_values.append(mar)
    
    #interpolate with our time axis
    if not mouth_timestamps:
        print(" C1: Warning - No valid mouth timestamps found.")
        state["lip_sync_score"] = 0.0
        return state
        
    mouth_signal = np.interp(time_axis, mouth_timestamps, mouth_mar_values)

    #get audio signal
    audio_signal = np.zeros_like(time_axis)
    for onset_time in audio_onsets:
        idx = np.searchsorted(time_axis, onset_time, side="left")
        if idx < len(audio_signal):
            audio_signal[idx] = 1.0
    
    #smooth signal using gaussian filters
    audio_signal = gaussian_filter1d(audio_signal, sigma=2)

    #normalize (v important)
    epsilon = 1e-9
    mouth_signal_norm = (mouth_signal - np.mean(mouth_signal)) / (np.std(mouth_signal) + epsilon)
    audio_signal_norm = (audio_signal - np.mean(audio_signal)) / (np.std(audio_signal) + epsilon)

    
    max_lag_frames = int(fps * 0.5)
    # Ensure signal is long enough
    if len(mouth_signal_norm) <= 2 * max_lag_frames:
         print(" C1: Warning - Signal too short for correlation.")
         state["lip_sync_score"] = 0.0
         return state

    sub_mouth_signal = mouth_signal_norm[max_lag_frames:-max_lag_frames]
    
    # `correlate` will check lags from -max_lag_frames to +max_lag_frames
    correlation = correlate(audio_signal_norm, sub_mouth_signal, mode='valid', method='fft')
    
    if len(sub_mouth_signal) > 0:
        normalized_correlation = correlation / len(sub_mouth_signal)
    else:
        normalized_correlation = np.array([0.0])

    # The lip-sync score is the maximum positive correlation found within the lag window.
    # We are interested in positive correlation (mouth opens when sound occurs).
    # We use np.max and clamp it at 0, as a negative correlation implies anti-sync.
    score = np.max(normalized_correlation)
    lip_sync_score = max(0.0, float(score))

    print(f" C1: Lip Sync Analysis Complete. Score: {lip_sync_score:.4f}")
    state["lip_sync_score"] = lip_sync_score

    return state