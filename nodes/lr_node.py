import json
import math
import os
from typing import Dict, Any
from nodes import dump_node_debug
import json
import os


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default


def _load_weights(path: str) -> Dict[str, float]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return {k: _safe_float(v, 0.0) for k, v in data.items()} if isinstance(data, dict) else {}
    except Exception:
        return {}

def _gesture_score_from_state(state: dict) -> float:
    gesture_checks = state.get("gesture_check", []) or []
    matched = sum(1 for g in gesture_checks if g.get("status") == "Consistent")
    missed = sum(1 for g in gesture_checks if g.get("status") == "Inconsistent")
    if (matched + missed) > 0:
        return matched / (matched + 0.5 * missed)

    data_dir = state.get("data_dir")
    if not data_dir:
        return 0.0
    debug_path = os.path.join(data_dir, "C2_debug.json")
    try:
        with open(debug_path, "r") as f:
            dbg = json.load(f)
        matched = _safe_float(dbg.get("matched"), 0.0)
        missed = _safe_float(dbg.get("inconsistent"), 0.0)
        if (matched + missed) > 0:
            return matched / (matched + 0.5 * missed)
    except Exception:
        pass
    return 0.0

def _headpose_jerk_from_state(pose_list: list) -> float:
    if len(pose_list) < 2:
        return 0.0
    diffs = []
    for prev, curr in zip(pose_list, pose_list[1:]):
        dt = _safe_float(curr.get("timestamp"), 0.0) - _safe_float(prev.get("timestamp"), 0.0)
        if dt <= 0:
            continue
        prev_pose = prev.get("pose", prev)
        curr_pose = curr.get("pose", curr)
        for key in ("yaw", "pitch", "roll"):
            diffs.append(abs(_safe_float(curr_pose.get(key), 0.0) - _safe_float(prev_pose.get(key), 0.0)) / dt)
    if not diffs:
        return 0.0
    return sum(diffs) / len(diffs)

def _texture_score_from_state(state: dict) -> float:
    score = _safe_float(state.get("texture_ela_score"), None)
    if score is not None:
        return score
    data_dir = state.get("data_dir")
    if not data_dir:
        return 0.0
    debug_path = os.path.join(data_dir, "V5_debug.json")
    try:
        with open(debug_path, "r") as f:
            dbg = json.load(f)
        val = dbg.get("avg_score")
        return _safe_float(val, 0.0)
    except Exception:
        return 0.0


def run(state: dict) -> dict:
    metadata = state.get("metadata", {})
    duration = _safe_float(metadata.get("duration"), 0.0)
    segments = state.get("segments", []) or []

    word_count = _safe_float(state.get("word_count"), 0.0)
    speech_rate = word_count / duration if duration > 0 else 0.0
    pause_total = 0.0
    if segments:
        sorted_segs = sorted(segments, key=lambda s: s.get("start", 0.0))
        for prev, curr in zip(sorted_segs, sorted_segs[1:]):
            gap = _safe_float(curr.get("start"), 0.0) - _safe_float(prev.get("end"), 0.0)
            if gap > 0:
                pause_total += gap
    pause_ratio = (pause_total / duration) if duration > 0 else 0.0

    lip_sync_score = _safe_float(state.get("lip_sync_score"), 0.0)

    gesture_score = _gesture_score_from_state(state)

    blink_data = state.get("blink_data", []) or []
    blink_rate = (len(blink_data) / duration) * 60.0 if duration > 0 else 0.0
    pose = state.get("head_pose_data", []) or []
    headpose_jerk = _headpose_jerk_from_state(pose)

    texture_score = _texture_score_from_state(state)
    claims = state.get("claims", []) or []
    evidence = state.get("evidence", []) or []
    supported_claims = [c for c in claims if _safe_float(c.get("evidence_score"), 0.0) > 0]
    avg_claim_reliability = 0.0
    if supported_claims:
        avg_claim_reliability = sum(_safe_float(c.get("evidence_score"), 0.0) for c in supported_claims) / len(
            supported_claims
        )
    evidence_avg = 0.0
    if evidence:
        evidence_avg = sum(_safe_float(e.get("reliability_score"), 0.0) for e in evidence) / max(len(evidence), 1)

    # Feature Normalization (Approximate to 0-1 range)
    # Blink rate: Normal ~15-30 bpm. Max expected ~60. 
    blink_rate_norm = min(blink_rate / 60.0, 1.0)
    
    # Headpose jerk: Normal < 10. Max expected ~100.
    headpose_jerk_norm = min(headpose_jerk / 100.0, 1.0)
    
    # Speech rate: Normal ~2.5 wps. Max expected ~5.
    speech_rate_norm = min(speech_rate / 5.0, 1.0)

    features: Dict[str, float] = {
        "speech_rate": speech_rate_norm,
        "pause_ratio": pause_ratio,
        "lip_sync": lip_sync_score,
        "gesture_score": gesture_score,
        "blink_rate": blink_rate_norm,
        "headpose_jerk": headpose_jerk_norm,
        "texture": texture_score,
        "claim_reliability": avg_claim_reliability,
        "evidence_reliability": evidence_avg,
    }

    weights = _load_weights("lr_weights.json")

    z = weights.get("bias", 0.0)
    print(f"LR Node: Bias = {z}")
    for k, v in features.items():
        w = weights.get(k, 0.0)
        contribution = w * v
        z += contribution
        print(f"LR Node: {k}: val={v}, weight={w}, contrib={contribution}")

    if z >= 0:
        fake_prob = 1.0 / (1.0 + math.exp(-z))
    else:
        fake_prob = math.exp(z) / (1.0 + math.exp(z))
    
    state["features"] = features
    state["fake_probability"] = fake_prob
    
    print(f"LR Node: Total z={z}, probability={fake_prob:.50f}")
    
    dump_node_debug(
        state,
        "LR",
        {"fake_probability": fake_prob, "features": features},
    )

    # Save features to cache
    try:
        input_path = state.get("input_path", "")
        if input_path:
            os.makedirs("features", exist_ok=True)
            video_name = os.path.basename(input_path)
            feature_file = os.path.join("features", f"{video_name}.json")
            with open(feature_file, "w") as f:
                json.dump(features, f, indent=2)
            print(f"LR Node: Saved features to {feature_file}")
    except Exception as e:
        print(f"LR Node: Warning - failed to save feature cache: {e}")

    label = state.get("label")
    if label in (0, 1):
        lr = 0.005
        error = fake_prob - float(label)
        weights["bias"] = weights.get("bias", 0.0) - lr * error * 1.0
        for k, v in features.items():
            weights[k] = weights.get(k, 0.0) - lr * error * v
        try:
            with open("lr_weights.json", "w") as f:
                json.dump(weights, f, indent=2)
        except Exception as e:
            print(f"LR: Warning - failed to save updated weights: {e}")

    return state
