import os
import cv2
import numpy as np


def _ela_score(image: np.ndarray, quality: int = 90) -> float:
    """Compute a lightweight Error Level Analysis score."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded = cv2.imencode(".jpg", image, encode_param)
    if not success:
        return 0.0
    reconstructed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if reconstructed is None or reconstructed.shape != image.shape:
        return 0.0
    diff = cv2.absdiff(image, reconstructed)
    return float(np.mean(diff))


def _texture_score(image: np.ndarray) -> float:
    """High-frequency texture score using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize to 0-1-ish range using a soft saturation
    return float(min(lap_var / 500.0, 1.0))


def run(state: dict) -> dict:
    print("Node V5: Running lightweight texture/ELA analysis...")

    keyframes = state.get("keyframes") or []
    debug = state.get("debug", False)

    if not keyframes:
        if debug:
            print("[DEBUG] V5: No keyframes available; skipping texture checks.")
        state["texture_anomaly_score"] = 0.0
        return state

    ela_scores = []
    tex_scores = []
    details = []

    # Limit processing to avoid excessive CPU use
    sample_paths = keyframes[: min(len(keyframes), 12)]

    for path in sample_paths:
        if not os.path.exists(path):
            if debug:
                print(f"[DEBUG] V5: Missing keyframe {path}")
            continue

        img = cv2.imread(path)
        if img is None:
            if debug:
                print(f"[DEBUG] V5: Failed to read {path}")
            continue

        ela = _ela_score(img)
        tex = _texture_score(img)

        ela_scores.append(ela)
        tex_scores.append(tex)
        details.append({"keyframe_path": path, "ela": ela, "texture": tex})

    if not ela_scores:
        print(" V5: No valid frames processed; defaulting texture score to 0.")
        state["texture_anomaly_score"] = 0.0
        return state

    # Higher ELA mean implies more compression inconsistencies -> suspicious
    ela_mean = float(np.mean(ela_scores))
    tex_mean = float(np.mean(tex_scores)) if tex_scores else 0.0

    # Convert to a 0-1 anomaly score; thresholds derived empirically for web-video
    ela_anomaly = min(ela_mean / 25.0, 1.0)
    texture_anomaly = 1.0 - min(tex_mean, 1.0)  # sharper textures -> lower anomaly

    combined = max(0.0, min((0.65 * ela_anomaly) + (0.35 * texture_anomaly), 1.0))

    state["texture_anomaly_score"] = combined
    state.setdefault("metadata", {})["texture_samples"] = len(details)
    state["texture_details"] = details

    if debug:
        print(f"[DEBUG] V5: ELA mean={ela_mean:.2f}, texture mean={tex_mean:.2f}, combined={combined:.2f}")

    return state
