from typing import List, Dict, Any
import math


def _get_speech_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    cleaned = []
    for seg in segments or []:
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            if end <= start:
                continue
            cleaned.append({"start": start, "end": end})
        except Exception:
            continue
    return cleaned


def _face_presence_timeline(face_detections: List[Dict[str, Any]]) -> Dict[int, bool]:
    presence = {}
    for det in face_detections or []:
        ts = det.get("timestamp")
        has_face = bool(det.get("faces"))
        if ts is None:
            continue
        bucket = int(math.floor(float(ts)))
        presence[bucket] = presence.get(bucket, False) or has_face
    return presence


def run(state: dict) -> dict:
    """
    Gesture/Narration heuristic:
    - Are faces present when speech occurs?
    - Is there at least one consistent on-camera speaker?
    - Outputs a normalized alignment score (1.0 = strong on-camera narration).
    """
    print("Node C2: Checking gesture/narration alignment...")

    segments = state.get("segments") or []
    face_detections = state.get("face_detections") or []

    speech_segments = _get_speech_segments(segments)
    face_timeline = _face_presence_timeline(face_detections)

    if not speech_segments:
        print(" C2: No speech segments available; skipping narration check.")
        state["narration_alignment"] = 0.0
        state["face_presence_ratio"] = 0.0
        return state

    total_speech = sum(seg["end"] - seg["start"] for seg in speech_segments)
    if total_speech == 0:
        state["narration_alignment"] = 0.0
        state["face_presence_ratio"] = 0.0
        return state

    # Calculate how often we have a face while speaking
    overlap_time = 0.0
    for seg in speech_segments:
        start_bucket = int(math.floor(seg["start"]))
        end_bucket = int(math.floor(seg["end"]))
        for bucket in range(start_bucket, end_bucket + 1):
            if face_timeline.get(bucket, False):
                overlap_time += 1.0

    # How often do we see any face at all?
    face_presence_ratio = (
        sum(1 for present in face_timeline.values() if present) /
        max(len(face_timeline), 1)
    )

    # Derive a narration alignment score: face present while talking matters most
    overlap_ratio = min(overlap_time / max(total_speech, 1e-6), 1.0)
    narration_alignment = max(0.0, min((0.7 * overlap_ratio) + (0.3 * face_presence_ratio), 1.0))

    state["narration_alignment"] = narration_alignment
    state["face_presence_ratio"] = face_presence_ratio

    if state.get("debug", False):
        print(f"[DEBUG] C2: Speech seconds={total_speech:.1f}, overlap={overlap_time:.1f}")
        print(f"[DEBUG] C2: Face presence ratio={face_presence_ratio:.2f}, alignment={narration_alignment:.2f}")

    return state
