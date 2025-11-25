import math
from statistics import mean


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _safe_mean(values, default: float = 0.0) -> float:
    filtered = [v for v in values if v is not None]
    return mean(filtered) if filtered else default


def run(state: dict) -> dict:
    """
    LR Node: Lightweight risk combiner.
    Converts upstream heuristic scores into a single fake_probability.
    """
    print("Node LR: Combining features into final fake probability...")

    lip_sync_score = _clamp(float(state.get("lip_sync_score", 0.0)))
    narration_alignment = _clamp(float(state.get("narration_alignment", 0.0)))
    texture_anomaly = _clamp(float(state.get("texture_anomaly_score", 0.0)))

    # Speech rhythm sanity: compare audio onsets vs transcript word count
    onset_count = state.get("onset_count") or 0
    word_count = state.get("word_count") or 0
    speech_risk = 0.0
    if onset_count and word_count:
        words_per_onset = word_count / max(onset_count, 1)
        # Expect roughly 1-4 words per onset; outside is suspicious
        if words_per_onset < 0.75:
            speech_risk = _clamp((0.75 - words_per_onset) / 0.75, 0.0, 1.0)
        elif words_per_onset > 4.0:
            speech_risk = _clamp((words_per_onset - 4.0) / 6.0, 0.0, 1.0)

    # Evidence confidence (higher = more likely real)
    evidence_scores = []
    for claim in state.get("claims") or []:
        if isinstance(claim, dict):
            evidence_scores.append(claim.get("evidence_score"))
    evidence_conf = _clamp(_safe_mean(evidence_scores, default=0.0))
    evidence_risk = 1.0 - evidence_conf

    # Blink/pose stability (optional)
    blink_data = state.get("blink_data") or []
    blink_rate = 0.0
    if blink_data:
        timestamps = [b.get("timestamp", 0.0) for b in blink_data]
        if timestamps:
            total_time = max(timestamps) - min(timestamps)
            if total_time > 0:
                blink_rate = len(blink_data) / total_time
                state["blink_rate"] = blink_rate
    blink_risk = 0.0
    if blink_rate:
        # Human blink rate ~0.1-0.4 per second
        if blink_rate < 0.05:
            blink_risk = _clamp((0.05 - blink_rate) / 0.05, 0.0, 1.0)
        elif blink_rate > 0.6:
            blink_risk = _clamp((blink_rate - 0.6) / 0.6, 0.0, 1.0)

    lip_sync_risk = 1.0 - lip_sync_score
    narration_risk = 1.0 - narration_alignment

    # Weighted blend; base_score to avoid returning 0.0 when data absent
    fake_probability = (
        0.30 * lip_sync_risk +
        0.20 * narration_risk +
        0.15 * texture_anomaly +
        0.15 * speech_risk +
        0.15 * evidence_risk +
        0.05 * blink_risk
    )

    fake_probability = _clamp(fake_probability)

    state["features"] = {
        "lip_sync_risk": round(lip_sync_risk, 3),
        "narration_risk": round(narration_risk, 3),
        "texture_anomaly": round(texture_anomaly, 3),
        "speech_risk": round(speech_risk, 3),
        "evidence_risk": round(evidence_risk, 3),
        "blink_risk": round(blink_risk, 3),
    }
    state["fake_probability"] = fake_probability

    if state.get("debug", False):
        print(f"[DEBUG] LR: Features -> {state['features']}")
        print(f"[DEBUG] LR: fake_probability = {fake_probability:.3f}")

    return state
