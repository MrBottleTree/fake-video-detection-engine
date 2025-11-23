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
    print("Node C1: Analyzing Lip Sync (Robust Windowed Correlation)...")

    mouth_landmarks = state.get("mouth_landmarks")
    audio_onsets = state.get("audio_onsets")
    metadata = state.get("metadata", {})
    fps = metadata.get("fps")
    duration = metadata.get("duration")

    if not mouth_landmarks or not audio_onsets:
        print(" C1: Warning - Missing mouth landmarks or audio onsets. Cannot compute lip-sync score.")
        state["lip_sync_score"] = 0.0 
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

    
    max_lag_frames = int(fps * 0.5) # +/- 0.5 seconds lag allowed
    
    # --- Robust Windowed Correlation ---
    # Instead of one global correlation, we compute correlation in overlapping windows
    # and take the average of the top N windows. This handles silence/noise.
    
    window_duration = 5.0 # seconds
    window_size = int(window_duration * fps)
    step_size = int(window_size / 2) # 50% overlap
    
    if len(mouth_signal_norm) < window_size:
        # Signal too short for windowing, fall back to global
        windows = [(mouth_signal_norm, audio_signal_norm)]
    else:
        windows = []
        for i in range(0, len(mouth_signal_norm) - window_size + 1, step_size):
            m_win = mouth_signal_norm[i : i + window_size]
            a_win = audio_signal_norm[i : i + window_size]
            windows.append((m_win, a_win))
            
    window_scores = []
    
    for m_win, a_win in windows:
        # Skip windows with no audio activity (silence)
        if np.std(a_win) < 0.01:
            continue
            
        # Skip windows with no mouth movement (static face)
        if np.std(m_win) < 0.01:
            continue
            
        # Cross-correlation
        # We only care about the center part of the correlation to avoid edge effects?
        # Actually, 'valid' mode handles this if we crop one signal.
        # But here both are same size. 'same' or 'full' is needed, or just standard correlate.
        # Let's use the same logic as before but on the window.
        
        # To allow lag, we need one signal to be shorter or just use full correlation and find peak near center.
        # Let's use full correlation.
        corr = correlate(a_win, m_win, mode='full')
        lags = np.arange(-len(a_win) + 1, len(a_win))
        
        # Filter for lags within max_lag_frames
        valid_indices = np.where(np.abs(lags) <= max_lag_frames)[0]
        if len(valid_indices) == 0:
            continue
            
        valid_corr = corr[valid_indices]
        
        # Normalize by length
        norm_corr = valid_corr / len(m_win)
        
        max_corr = np.max(norm_corr)
        window_scores.append(max_corr)
        
    if not window_scores:
        print(" C1: Warning - No valid windows found (silence or static). Defaulting to 0.0")
        lip_sync_score = 0.0
    else:
        # Take the average of the top 50% of window scores
        # This assumes at least half the video should be in sync if it's real/good fake.
        # It ignores bad segments (looking away, silence).
        window_scores.sort(reverse=True)
        top_n = max(1, int(len(window_scores) * 0.5))
        top_scores = window_scores[:top_n]
        lip_sync_score = float(np.mean(top_scores))
        
        # Clamp
        lip_sync_score = max(0.0, lip_sync_score)

    print(f" C1: Lip Sync Analysis Complete. Score: {lip_sync_score:.4f} (Computed over {len(window_scores)} windows)")
    state["lip_sync_score"] = lip_sync_score

    return state