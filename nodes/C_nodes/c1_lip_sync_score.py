import numpy as np
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d
from typing import TypedDict, Optional, Dict, Any, Annotated
import operator

def run(state: dict) -> dict:
    print(" C1: Analyzing Lip Sync...")

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

    #use mouth aspect ratio for getting mouth movements
    mouth_timestamps = [lm['timestamp'] for lm in mouth_landmarks]
    mouth_mar_values = [lm['mar'] for lm in mouth_landmarks]
    
    #interpolate with our time axis
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
    sub_mouth_signal = mouth_signal_norm[max_lag_frames:-max_lag_frames]
    
    # `correlate` will check lags from -max_lag_frames to +max_lag_frames
    correlation = correlate(audio_signal_norm, sub_mouth_signal, mode='valid', method='fft')
    
    normalized_correlation = correlation / len(sub_mouth_signal)

    # The lip-sync score is the maximum positive correlation found within the lag window.
    # We are interested in positive correlation (mouth opens when sound occurs).
    # We use np.max and clamp it at 0, as a negative correlation implies anti-sync.
    score = np.max(normalized_correlation)
    lip_sync_score = max(0.0, float(score))

    print(f" C1: Lip Sync Analysis Complete. Score: {lip_sync_score:.4f}")
    state["lip_sync_score"] = lip_sync_score

    return state