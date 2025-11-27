import streamlit as st
import os
import sys
import time
import json
import io
from PIL import Image
from main import app as graph_app
from main import State

# Page Config
st.set_page_config(
    page_title="Deepfake Detection Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for "Professional/Glass" Look ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Card Styling */
    .css-1r6slb0, .css-12oz5g7 {
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 20px;
    }
    
    /* Verdict Box */
    .verdict-box {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .verdict-fake {
        background: linear-gradient(135deg, #3a0000 0%, #8a0000 100%);
        border: 2px solid #ff4b4b;
        color: #ffcccc;
    }
    .verdict-real {
        background: linear-gradient(135deg, #002a00 0%, #006400 100%);
        border: 2px solid #00cc00;
        color: #ccffcc;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #00c6ff, #0072ff);
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/security-checked.png", width=60)
    st.title("Detector Config")
    
    st.markdown("### 📥 Input Source")
    # Check for pre-filled video URL from environment variable
    default_url = os.environ.get("STREAMLIT_VIDEO_URL", "https://www.youtube.com/watch?v=example")
    input_url = st.text_input("Video URL", default_url, help="Enter a YouTube URL or direct video link.")
    
    st.markdown("### ⚙️ System Settings")
    # Check for debug mode from environment variable
    default_debug = os.environ.get("STREAMLIT_DEBUG_MODE", "0") == "1"
    debug_mode = st.toggle("Debug Mode", value=default_debug)
    show_graph = st.toggle("Show Graph Architecture", value=True)
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    status_indicator = st.empty()
    status_indicator.success("System Ready")
    
    if st.button("🧹 Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("v2.0.0 | Agentic AI Engine")

# --- Main Content ---
st.title("🛡️ Deepfake Detection Engine")
st.markdown("#### Advanced Multi-Modal Verification System")

# Session State Initialization
if "processing" not in st.session_state:
    st.session_state.processing = False
if "result_state" not in st.session_state:
    st.session_state.result_state = None
if "graph_image" not in st.session_state:
    try:
        st.session_state.graph_image = graph_app.get_graph().draw_mermaid_png()
    except Exception as e:
        st.warning(f"Could not render graph visualization: {e}")
        st.session_state.graph_image = None

# --- Graph Architecture View ---
if show_graph and st.session_state.graph_image:
    with st.expander("🕸️ Execution Graph Architecture", expanded=False):
        st.image(st.session_state.graph_image, caption="LangGraph Execution Flow", use_column_width=True)

# --- Execution Logic ---
def run_analysis():
    st.session_state.processing = True
    st.session_state.result_state = None
    status_indicator.info("Processing...")
    
    # Initial State
    initial_state = {
        "input_path": input_url,
        "debug": debug_mode
    }
    
    # Layout for Progress
    col1, col2 = st.columns([2, 1])
    
    with col1:
        progress_bar = st.progress(0)
        current_stage = st.empty()
    
    with col2:
        node_status = st.empty()
    
    # Log Container
    log_expander = st.expander("📜 Live Execution Logs", expanded=True)
    
    # Nodes list for progress tracking
    nodes_order = ["IN", "A1", "V1", "A2", "A3", "V2", "V3", "V4", "V5", "C1", "C2", "C3", "E1", "E2", "E3", "LR"]
    total_steps = len(nodes_order)
    current_step = 0
    final_state = None
    
    try:
        with log_expander:
            st.code("Initializing Graph Execution...", language="bash")
            
            # Stream execution
            for output in graph_app.stream(initial_state):
                for node_name, state_update in output.items():
                    current_step += 1
                    progress = min(current_step / total_steps, 1.0)
                    progress_bar.progress(progress)
                    
                    # Update Status
                    current_stage.markdown(f"**Executing Node:** `{node_name}`")
                    node_status.success(f"Completed: {node_name}")
                    
                    # Log specific interesting events
                    if "claims" in state_update:
                        count = len(state_update['claims'])
                        st.write(f"ℹ️ **{node_name}**: Extracted {count} claims.")
                    if "fake_probability" in state_update:
                        prob = state_update['fake_probability']
                        st.write(f"⚠️ **{node_name}**: Intermediate Fake Prob: {prob:.2%}")
                        
                    final_state = state_update
                    
            progress_bar.progress(1.0)
            current_stage.markdown("**✅ Analysis Complete**")
            status_indicator.success("Analysis Complete")
            
    except Exception as e:
        st.error(f"❌ Critical Error during execution: {e}")
        st.session_state.processing = False
        status_indicator.error("System Error")
        return

    st.session_state.processing = False
    st.session_state.result_state = final_state

# --- Action Button ---
if st.button("🚀 Start Deepfake Analysis", disabled=st.session_state.processing, use_container_width=True):
    if not input_url:
        st.warning("Please enter a valid Video URL.")
    else:
        run_analysis()

# --- Results Dashboard ---
if st.session_state.result_state:
    state = st.session_state.result_state
    
    st.markdown("---")
    st.subheader("📊 Analysis Report")
    
    # 1. Verdict Section
    fake_prob = state.get("fake_probability", 0.0)
    verdict = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = fake_prob if verdict == "FAKE" else (1 - fake_prob)
    color_class = "verdict-fake" if verdict == "FAKE" else "verdict-real"
    
    st.markdown(f"""
    <div class="verdict-box {color_class}">
        <h1 style="margin:0; font-size: 3em;">{verdict}</h1>
        <h3 style="margin:0; opacity: 0.8;">Confidence: {confidence:.1%}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Detailed Tabs
    tab_video, tab_claims, tab_bio, tab_data = st.tabs([
        "📺 Video Analysis", 
        "⚖️ Claims & Evidence", 
        "🧬 Biometrics", 
        "💾 Raw Data"
    ])
    
    with tab_video:
        c1, c2 = st.columns([3, 2])
        with c1:
            if state.get("data_dir"):
                video_path = os.path.join(state["data_dir"], "video.mp4")
                if os.path.exists(video_path):
                    st.video(video_path)
                else:
                    st.warning("Video file missing from processed directory.")
        with c2:
            st.info("Metadata")
            meta = state.get("metadata", {})
            st.dataframe(meta, use_container_width=True)
            
    with tab_claims:
        claims = state.get("claims", [])
        if claims:
            st.success(f"Found {len(claims)} factual claims.")
            for i, c in enumerate(claims):
                with st.expander(f"Claim #{i+1}: {c.get('verdict', 'Unknown')} ({c.get('evidence_score', 0):.2f})"):
                    st.markdown(f"**Claim:** *\"{c.get('text')}\"*")
                    
                    # Evidence
                    evidence = state.get("evidence", [])
                    related_ev = [e for e in evidence if e.get("claim_text") == c.get("text")]
                    
                    if related_ev:
                        st.markdown("---")
                        st.markdown("**🔍 Evidence Sources:**")
                        for rev in related_ev:
                            rel_score = rev.get('reliability_score', 0)
                            col_a, col_b = st.columns([4, 1])
                            with col_a:
                                st.markdown(f"[{rev.get('title', 'Source Link')}]({rev.get('url')})")
                                st.caption(f"\"{rev.get('snippet', '')[:150]}...\"")
                            with col_b:
                                st.metric("Reliability", f"{rel_score:.2f}")
                    else:
                        st.warning("No direct evidence found for this claim.")
        else:
            st.info("No factual claims were extracted from the audio.")
            
    with tab_bio:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Lip Sync Error", f"{state.get('lip_sync_score', 0):.4f}", help="Lower is better (usually)")
        with c2:
            st.metric("Word Count", state.get("word_count", 0))
        with c3:
            st.metric("Detected Faces", len(state.get("face_detections", []) or []))
            
    with tab_data:
        st.json(state)
