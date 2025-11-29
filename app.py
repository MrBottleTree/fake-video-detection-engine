import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st
from graphviz import Digraph

from main import app as graph_app

# --- Page Setup & Styling ---
st.set_page_config(
    page_title="Deepfake Detection Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        body, .stApp { font-family: "Inter", "Helvetica", sans-serif; }
        .status-pill { border-radius: 10px; padding: 6px 8px; border: 1px solid #e5e7eb; background: #f9fafb; font-size: 12px; }
        .verdict-card { padding: 14px; border: 1px solid #e5e7eb; border-radius: 10px; background: #f8fafc; text-align: center; }
        .status-table { width: 100%; font-size: 13px; border-collapse: collapse; }
        .status-table th, .status-table td { padding: 6px 8px; border-bottom: 1px solid #edf0f3; text-align: left; }
        .status-badge { padding: 3px 8px; border-radius: 999px; font-size: 11px; display: inline-block; }
        .badge-queued { background: #f3f4f6; color: #374151; }
        .badge-running { background: #eef2ff; color: #4338ca; }
        .badge-done { background: #ecfdf3; color: #166534; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Helpers ---
def load_graph_image() -> bytes | None:
    if "graph_image" in st.session_state:
        return st.session_state.graph_image
    try:
        img = graph_app.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)
    except Exception as e:  # pragma: no cover - visualization best-effort
        st.sidebar.warning(f"Could not render graph visualization: {e}")
        img = None
    st.session_state.graph_image = img
    return img


def resolve_input_path(source_choice: str, url_value: str, uploaded) -> Tuple[str | None, str | None]:
    if source_choice == "Upload":
        if uploaded is None:
            return None, "Upload a video to proceed."
        tmp_dir = Path(tempfile.mkdtemp(prefix="df_video_"))
        tmp_path = tmp_dir / uploaded.name
        tmp_path.write_bytes(uploaded.getvalue())
        return str(tmp_path), f"Uploaded file saved to {tmp_path}"
    if not url_value:
        return None, "Enter a video URL."
    return url_value, None


def render_status_board(container, statuses: Dict[str, str], order: List[str], labels: Dict[str, str], outputs: Dict[str, str]):
    badge_class = {"queued": "badge-queued", "running": "badge-running", "done": "badge-done"}
    rows = []
    for node in order:
        status = statuses.get(node, "queued")
        rows.append(
            f"<tr><td>{node}</td>"
            f"<td><span class='status-badge {badge_class.get(status,'badge-queued')}'>{status}</span></td>"
            f"<td>{labels.get(node, node)} — {outputs.get(node, 'waiting...')}</td></tr>"
        )
    html = "<table class='status-table'><tr><th>Node</th><th>Status</th><th>Output</th></tr>" + "".join(rows) + "</table>"
    container.markdown(html, unsafe_allow_html=True)


def safe_get(state: Dict[str, Any], key: str, default: Any = None):
    return state.get(key, default) if state else default


def render_graph(statuses: Dict[str, str], node_outputs: Dict[str, str]) -> Digraph:
    colors = {"queued": "#374151", "running": "#fbbf24", "done": "#10b981"}
    g = Digraph(engine="dot")
    g.attr(rankdir="LR", bgcolor="transparent")
    for node, label in node_labels.items():
        status = statuses.get(node, "queued")
        body = node_outputs.get(node, "")
        node_label = f"{node}\\n{label}"
        if body:
            node_label += f"\\n{body}"
        g.node(
            node,
            label=node_label,
            style="filled,rounded",
            shape="box",
            fillcolor=colors.get(status, "#374151"),
            fontcolor="#0b1021" if status == "done" else "#e5e7eb",
            color="#1f2937",
        )
    for src, dst in edges:
        g.edge(src, dst, color="#6b7280")
    return g


def update_node_outputs(node_name: str, state_update: Dict[str, Any], outputs: Dict[str, str]):
    def fmt(val, prec=2):
        try:
            return f"{float(val):.{prec}f}"
        except Exception:
            return str(val)

    try:
        if node_name == "A2":
            outputs[node_name] = f"words {state_update.get('word_count', 0)}"
        elif node_name == "V1":
            outputs[node_name] = f"keyframes {len(state_update.get('keyframes', []) or [])}"
        elif node_name == "V2":
            detections = state_update.get("ocr_results", []) or []
            outputs[node_name] = f"OCR {len(detections)} frames"
        elif node_name == "V3":
            outputs[node_name] = f"frames {len(state_update.get('mouth_landmarks', []) or [])}"
        elif node_name == "V4":
            outputs[node_name] = f"blink {len(state_update.get('blink_data', []) or [])}"
        elif node_name == "V5":
            outputs[node_name] = f"ELA {fmt(state_update.get('texture_ela_score', 0))}"
        elif node_name == "C1":
            outputs[node_name] = f"lip {fmt(state_update.get('lip_sync_score', 0))}"
        elif node_name == "C3":
            outputs[node_name] = f"claims {len(state_update.get('claims', []) or [])}"
        elif node_name == "E3":
            outputs[node_name] = "scored"
        elif node_name == "LR":
            fp = state_update.get("fake_probability")
            if fp is not None:
                outputs[node_name] = f"fake {fmt(fp)}"
    except Exception:
        pass


# --- Sidebar ---
with st.sidebar:
    st.title("Detector Controls")

    input_mode = st.radio("Input Type", ["URL", "Upload"], horizontal=True)
    env_default_url = os.environ.get("STREAMLIT_VIDEO_URL", "")
    url_input = st.text_input(
        "Video URL",
        env_default_url,
        help="YouTube link or direct video file URL.",
        disabled=input_mode == "Upload",
    )
    uploaded_file = st.file_uploader(
        "Upload video",
        type=["mp4", "mov", "mkv", "webm"],
        disabled=input_mode == "URL",
        help="Local video file for analysis.",
    )

    default_debug = os.environ.get("STREAMLIT_DEBUG_MODE", "0") == "1"
    debug_mode = st.toggle("Debug mode", value=default_debug)
    show_graph = st.toggle("Show pipeline map", value=True)

    st.markdown("---")
    st.caption("Deepfake Detection Engine · v2.1")


st.title("Deepfake Detection Engine")
st.caption("Lean control: start, watch the graph, and track node outputs in the table.")


# --- Run button & progress ---
if "processing" not in st.session_state:
    st.session_state.processing = False
if "result_state" not in st.session_state:
    st.session_state.result_state = None

node_labels = {
    "IN": "Load video",
    "A1": "Audio standardize",
    "A2": "ASR (Whisper)",
    "A3": "Audio events",
    "V1": "Keyframes & faces",
    "V2": "OCR overlays",
    "V3": "Mouth landmarks",
    "V4": "Blink & pose",
    "V5": "Texture/ELA",
    "C1": "Lip sync",
    "C2": "Gesture alignment",
    "C3": "Claim extraction",
    "E1": "Evidence search",
    "E2": "Source reliability",
    "E3": "Claim scoring",
    "LR": "Ensemble score",
}
nodes_order = list(node_labels.keys())

edges = [
    ("IN", "A1"),
    ("IN", "V1"),
    ("A1", "A2"),
    ("A1", "A3"),
    ("V1", "V2"),
    ("V1", "V3"),
    ("V1", "V4"),
    ("V1", "V5"),
    ("A3", "C1"),
    ("V3", "C1"),
    ("A2", "C2"),
    ("V1", "C2"),
    ("A2", "C3"),
    ("V2", "C3"),
    ("C3", "E1"),
    ("E1", "E2"),
    ("E2", "E3"),
    ("C3", "E3"),
    ("E1", "E3"),
    ("C1", "LR"),
    ("C2", "LR"),
    ("V4", "LR"),
    ("V5", "LR"),
    ("E3", "LR"),
]


def run_analysis(input_path: str):
    st.session_state.processing = True
    st.session_state.result_state = None

    progress_bar = st.progress(0)
    stage_text = st.empty()
    graph_box = st.empty()
    status_table_box = st.empty()

    state_snapshot: Dict[str, Any] = {}
    node_outputs: Dict[str, str] = {}
    
    # Initial status: Roots are running, others queued
    initial_statuses = {}
    # Build dependency map first
    node_parents = {}
    for src, dst in edges:
        node_parents.setdefault(dst, set()).add(src)

    for n in nodes_order:
        parents = node_parents.get(n, set())
        if not parents:
            initial_statuses[n] = "running"
        else:
            initial_statuses[n] = "queued"

    render_status_board(status_table_box, initial_statuses, nodes_order, node_labels, node_outputs)
    total_steps = len(nodes_order)
    done_nodes = set()

    # Build dependency map to prevent premature status updates
    # (Moved up)

    try:
        for output in graph_app.stream({"input_path": input_path, "debug": debug_mode}):
            current_batch = set(output.keys())
            
            # Filter out nodes that ran early (dependencies not met)
            valid_batch = set()
            for node in current_batch:
                parents = node_parents.get(node, set())
                # Only mark as valid if all parents are done
                if parents.issubset(done_nodes):
                    valid_batch.add(node)

            if valid_batch:
                stage_text.markdown("**Executing:** " + ", ".join(f"`{n}`" for n in valid_batch))

            for node_name, state_update in output.items():
                if node_name not in valid_batch:
                    continue
                state_snapshot.update(state_update)
                node_outputs.setdefault(node_name, "running")
                update_node_outputs(node_name, state_update, node_outputs)

            # Update done nodes immediately to trigger next steps
            done_nodes.update(valid_batch)

            statuses = {}
            for n in nodes_order:
                if n in done_nodes:
                    statuses[n] = "done"
                    continue
                
                # Check if all parents are done
                parents = node_parents.get(n, set())
                if not parents:
                    # Root nodes are running if not done
                    statuses[n] = "running"
                elif parents.issubset(done_nodes):
                    # Dependencies met -> Running
                    statuses[n] = "running"
                else:
                    statuses[n] = "queued"

            render_status_board(status_table_box, statuses, nodes_order, node_labels, node_outputs)
            progress_bar.progress(len(done_nodes) / total_steps)
            graph = render_graph(statuses, node_outputs)
            graph_box.graphviz_chart(graph, width="stretch")
            progress_bar.progress(len(done_nodes) / total_steps)

        stage_text.markdown("**Analysis complete**")
        progress_bar.progress(1.0)
        st.success("Processing finished.")
        st.session_state.result_state = state_snapshot

    except Exception as e:  # pragma: no cover - runtime safety
        st.error(f"Critical error during execution: {e}")
    finally:
        st.session_state.processing = False


if st.button("Start Deepfake Analysis", type="primary", disabled=st.session_state.processing):
    resolved_path, err = resolve_input_path(input_mode, url_input, uploaded_file)
    if err:
        st.warning(err)
    else:
        run_analysis(resolved_path)


# --- Pipeline map ---
# Pre-run static graph removed to keep the view minimal and avoid stale visuals.


# --- Results ---
if st.session_state.result_state:
    state = st.session_state.result_state
    meta = safe_get(state, "metadata", {})
    fake_prob = float(safe_get(state, "fake_probability", 0.0) or 0.0)
    verdict = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = fake_prob if verdict == "FAKE" else (1 - fake_prob)
    verdict_class = "verdict-fake" if verdict == "FAKE" else "verdict-real"

    st.markdown("---")
    st.subheader("Analysis Report")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown(
            f"""
            <div class="verdict-card {verdict_class}">
                <h1>{verdict}</h1>
                <h3>Confidence: {confidence:.1%}</h3>
                <p style="opacity:0.8;margin-top:8px;">Lower is better for authenticity.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.metric("Duration (s)", f"{meta.get('duration', 0):.2f}")
        st.metric("FPS", f"{meta.get('fps', 0):.2f}")
    with c3:
        st.metric("Faces detected", len(state.get("face_detections", []) or []))
        st.metric("Claims extracted", len(state.get("claims", []) or []))

    st.markdown("### Key Outputs")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Word count", state.get("word_count", 0))
    m2.metric("Lip-sync score", f"{state.get('lip_sync_score', 0):.4f}")
    m3.metric("Audio onsets", state.get("onset_count", 0))
    m4.metric("Texture/ELA score", f"{state.get('texture_ela_score', 0.0):.2f}")

    tabs = st.tabs(
        [
            "Media",
            "Frames & Faces",
            "Claims & Evidence",
            "Text & OCR",
            "Signals",
            "Raw State",
        ]
    )

    # Media tab
    with tabs[0]:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            video_path = None
            if state.get("data_dir"):
                candidate = Path(state["data_dir"]) / "video.mp4"
                if candidate.exists():
                    video_path = str(candidate)
            if video_path:
                st.video(video_path)
            else:
                st.warning("Processed video not found.")
        with col_b:
            hp = Path(state.get("data_dir", ".")) / "headpose_viz.mp4"
            lm = Path(state.get("data_dir", ".")) / "landmarks_viz.mp4"
            if hp.exists():
                st.video(str(hp), format="video/mp4")
            if lm.exists():
                st.video(str(lm), format="video/mp4")
            st.write("Metadata")
            st.json(meta, expanded=False)

    # Frames & Faces tab
    with tabs[1]:
        keyframes = state.get("keyframes", []) or []
        faces = [f["faces"][0]["crop_path"] for f in state.get("face_detections", []) if f.get("faces")]

        st.markdown("**Keyframes (sample)**")
        if keyframes:
            cols = st.columns(5)
            for idx, kf in enumerate(keyframes[:10]):
                with cols[idx % 5]:
                    st.image(kf, width="stretch")
        else:
            st.info("No keyframes available.")

        st.markdown("**Detected Faces (sample)**")
        if faces:
            cols = st.columns(5)
            for idx, face_path in enumerate(faces[:10]):
                with cols[idx % 5]:
                    st.image(face_path, width="stretch")
        else:
            st.info("No face crops available.")

    # Claims & Evidence tab
    with tabs[2]:
        claims = state.get("claims", []) or []
        evidence = state.get("evidence", []) or []
        if not claims:
            st.info("No factual claims extracted.")
        else:
            for i, claim in enumerate(claims):
                with st.expander(f"Claim #{i+1}: {claim.get('text','')[:80]}"):
                    st.write(f"**Claim:** {claim.get('text','')}")
                    st.write(f"**Verdict:** {claim.get('verdict','Unknown')} (score {claim.get('evidence_score',0):.2f})")
                    related = [ev for ev in evidence if ev.get("claim_text") == claim.get("text")]
                    if related:
                        st.markdown("**Evidence**")
                        for ev in related[:5]:
                            rel = ev.get("reliability_score", 0)
                            st.markdown(f"- [{ev.get('title','Source')}]({ev.get('url')}) · Reliability: {rel:.2f}")
                            if ev.get("snippet"):
                                st.caption(ev["snippet"][:180] + "...")
                    else:
                        st.info("No evidence matched this claim.")

    # Text & OCR tab
    with tabs[3]:
        transcript = state.get("transcript", "")
        segments = state.get("segments", []) or []
        ocr_results = state.get("ocr_results", []) or []

        st.markdown("**Transcript**")
        if transcript:
            st.text_area("Full transcript", transcript, height=220)
        else:
            st.info("No transcript available.")

        st.markdown("**ASR Segments (first 5)**")
        if segments:
            st.json(segments[:5], expanded=False)
        else:
            st.caption("No segments to display.")

        st.markdown("**OCR Extractions (first 5)**")
        if ocr_results:
            for item in ocr_results[:5]:
                st.write(f"- {item.get('text','')}")
        else:
            st.caption("No OCR text detected.")

    # Signals tab
    with tabs[4]:
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Lip-sync score", f"{state.get('lip_sync_score', 0):.4f}")
        col_s2.metric("Blink samples", len(state.get("blink_data", []) or []))
        col_s3.metric("Pose samples", len(state.get("head_pose_data", []) or []))

        st.markdown("**Audio envelope (length)**")
        envelope = state.get("audio_envelope", [])
        st.write(len(envelope))

        st.markdown("**Onsets**")
        st.write(state.get("audio_onsets", [])[:20])

    # Raw State tab
    with tabs[5]:
        st.json(state, expanded=False)
