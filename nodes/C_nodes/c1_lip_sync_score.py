import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import distance as dist
import os

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Node C1: Analyzing Lip Sync (Robust Correlation) on {device.upper()}...")

    mouth_landmarks = state.get("mouth_landmarks")
    metadata = state.get("metadata", {})
    fps = metadata.get("fps")
    duration = metadata.get("duration")
    
    # For testing, allow injecting signals directly
    test_audio_signal = state.get("test_audio_signal")
    # New: Get pre-calculated envelope from A3 node
    audio_envelope = state.get("audio_envelope")

    if not mouth_landmarks:
        print(" C1: Warning - Missing mouth landmarks. Cannot compute lip-sync score.")
        state["lip_sync_score"] = 0.0 
        return state

    if not fps or not duration:
        print(" C1: Warning - Missing video FPS or duration. Cannot compute lip-sync score.")
        state["lip_sync_score"] = 0.0
        return state

    # 1. Get Visual Signal (MAR)
    num_frames = int(duration * fps)
    time_axis = np.linspace(0, duration, num_frames)
    
    mouth_timestamps = []
    mouth_mar_values = []
    
    for lm in mouth_landmarks:
        timestamp = lm.get('timestamp', 0.0)
        if 'mar' in lm:
            mar = lm['mar']
        elif 'landmarks' in lm:
            points = lm['landmarks']
            mar = calculate_mar(points) if points else 0.0
        else:
            mar = 0.0
        mouth_timestamps.append(timestamp)
        mouth_mar_values.append(mar)
    
    if not mouth_timestamps:
        print(" C1: Warning - No valid mouth timestamps.")
        state["lip_sync_score"] = 0.0
        return state
        
    # Interpolate MAR to constant FPS
    mouth_signal = np.interp(time_axis, mouth_timestamps, mouth_mar_values)

    # 2. Get Audio Signal (RMS)
    audio_signal = None
    if test_audio_signal is not None:
        audio_signal = np.array(test_audio_signal)
    elif audio_envelope is not None:
        audio_signal = np.array(audio_envelope)
    
    if audio_signal is None:
        print(" C1: Warning - Could not obtain audio signal (missing 'audio_envelope'). Defaulting to 0.0")
        state["lip_sync_score"] = 0.0
        return state
        
    # Resize if needed (robustness against slight frame count mismatches)
    if len(audio_signal) != len(mouth_signal):
         audio_signal = np.interp(
            np.linspace(0, 1, len(mouth_signal)),
            np.linspace(0, 1, len(audio_signal)),
            audio_signal
        )

    # 3. Normalize Signals
    epsilon = 1e-9
    mouth_signal_norm = (mouth_signal - np.mean(mouth_signal)) / (np.std(mouth_signal) + epsilon)
    audio_signal_norm = (audio_signal - np.mean(audio_signal)) / (np.std(audio_signal) + epsilon)

    # 4. GPU Acceleration with PyTorch
    try:
        # Convert to tensors
        m_tensor = torch.tensor(mouth_signal_norm, dtype=torch.float32, device=device).view(1, 1, -1)
        a_tensor = torch.tensor(audio_signal_norm, dtype=torch.float32, device=device).view(1, 1, -1)
        
        # Window parameters
        window_duration = 5.0 # seconds
        window_size = int(window_duration * fps)
        step_size = int(window_size / 2)
        
        if m_tensor.shape[2] < window_size:
            windows_m = [m_tensor]
            windows_a = [a_tensor]
        else:
            windows_m = m_tensor.unfold(2, window_size, step_size).squeeze(0).permute(1, 0, 2) # (N_win, 1, W)
            windows_a = a_tensor.unfold(2, window_size, step_size).squeeze(0).permute(1, 0, 2) # (N_win, 1, W)

        scores = []
        max_lag = int(fps * 0.5) # +/- 0.5s
        
        # Process windows
        for i in range(windows_m.shape[0]):
            w_m = windows_m[i] # (1, W)
            w_a = windows_a[i] # (1, W)
            
            # Skip silence/static
            if torch.std(w_a) < 0.01 or torch.std(w_m) < 0.01:
                continue
                
            # Cross correlation
            # We want 'same' or 'valid' with padding.
            # conv1d(input, weight)
            # input: w_a (1, 1, W)
            # weight: w_m (1, 1, W)
            # This gives a single scalar if no padding.
            # We want lags.
            # Let's pad audio window.
            
            pad_amount = max_lag
            w_a_padded = F.pad(w_a.unsqueeze(0), (pad_amount, pad_amount)) # (1, 1, W + 2*lag)
            w_m_kernel = w_m.unsqueeze(0) # (1, 1, W)
            
            # Cross-corr via conv1d
            # Output size: (W + 2*lag) - W + 1 = 2*lag + 1
            cc = F.conv1d(w_a_padded, w_m_kernel) # (1, 1, 2*lag+1)
            cc = cc.squeeze() / window_size # Normalize by length
            
            max_corr = torch.max(cc).item()
            
            # Peak Sharpness: Max / Mean (of absolute correlations)
            # High sharpness = good sync. Low sharpness (flat) = bad sync.
            mean_corr = torch.mean(torch.abs(cc)).item() + epsilon
            sharpness = max_corr / mean_corr
            
            # Combined score: correlation weighted by sharpness?
            # Or just correlation if it's high enough.
            # Let's use max_corr but penalize if sharpness is low.
            
            final_score = max_corr
            if sharpness < 1.5: # Arbitrary threshold for "flat" peak
                final_score *= 0.5
                
            scores.append(final_score)
            
        if not scores:
            print(" C1: Warning - No valid windows. Defaulting to 0.0")
            lip_sync_score = 0.0
        else:
            # Average of top 50% scores
            scores.sort(reverse=True)
            top_n = max(1, int(len(scores) * 0.5))
            lip_sync_score = float(np.mean(scores[:top_n]))
            lip_sync_score = max(0.0, lip_sync_score)

    except Exception as e:
        print(f" C1: Error during GPU/Tensor processing: {e}. Falling back to CPU/NumPy.")
        # Fallback logic could go here, but for now we assume torch works.
        lip_sync_score = 0.0

    print(f" C1: Lip Sync Analysis Complete. Score: {lip_sync_score:.4f}")
    state["lip_sync_score"] = lip_sync_score
    return state