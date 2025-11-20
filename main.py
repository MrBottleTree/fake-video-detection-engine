from langgraph.graph import StateGraph, END
from nodes import *

class State:
    input_path: str
    label: int

def in_node(state: State) -> State:
    return state

graph = StateGraph(State)

graph.add_node("IN", in_node)

graph.add_node("A1", a1_demux_audio_extract.run)
graph.add_node("A2", a2_vad_asr.run)
graph.add_node("A3", a3_audio_onsets.run)

graph.add_node("V1", v1_keyframes_facetrack.run)
graph.add_node("V2", v2_ocr_overlays.run)
graph.add_node("V3", v3_mouth_landmarks_timeseries.run)
graph.add_node("V4", v4_blink_headpose_dynamics.run)
graph.add_node("V5", v5_texture_ela.run)

graph.add_node("C1", c1_lip_sync_score.run)
graph.add_node("C2", c2_gesture_narration_check.run)
graph.add_node("C3", c3_claim_extraction.run)

graph.add_node("E1", e1_web_evidence.run)
graph.add_node("E2", e2_source_reliability.run)
graph.add_node("E3", e3_claim_evidence_scorer.run)

graph.add_node("LR", lr_node.run)

graph.set_entry_point("IN")
graph.add_edge("IN", "A1")
graph.add_edge("IN", "V1")
graph.add_edge("A1", "A2")
graph.add_edge("A1", "A3")
graph.add_edge("V1", "V2")
graph.add_edge("V1", "V3")
graph.add_edge("V1", "V4")
graph.add_edge("V1", "V5")
graph.add_edge("A3", "C1")
graph.add_edge("V3", "C1")
graph.add_edge("A2", "C2")
graph.add_edge("V1", "C2")
graph.add_edge("A2", "C3")
graph.add_edge("V2", "C3")
graph.add_edge("C3", "E1")
graph.add_edge("E1", "E2")
graph.add_edge("E2", "E3")
graph.add_edge("C1", "LR")
graph.add_edge("C2", "LR")
graph.add_edge("V4", "LR")
graph.add_edge("V5", "LR")
graph.add_edge("E3", "LR")

graph.add_edge("LR", END)

app = graph.compile()

def main() -> None:
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py <video_or_url> [label_0_or_1]")
        raise sys.exit(1)

    input_path = sys.argv[1]
    state: State = {"input_path": input_path}

    if len(sys.argv) >= 3:
        try:
            state["label"] = int(sys.argv[2])
        except ValueError:
            print("Label must be 0 or 1 if provided.")
            raise sys.exit(1)

    result = app.invoke(state)

    fake_prob = result.get("fake_probability")
    print("Final state:", result)
    print("Fake probability:", fake_prob)


if __name__ == "__main__":
    main()
