import os
import whisper
import torch
import imageio_ffmpeg
import warnings

# Suppress FP16 warning on CPU
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

def run(state: dict) -> dict:
    print("Node A2: Running VAD and ASR...")
    
    # Ensure ffmpeg is in PATH for whisper
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_path)
    if ffmpeg_dir not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + ffmpeg_dir

    audio_path = state.get("audio_path")
    
    if not audio_path or not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return state

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        model = whisper.load_model("base", device=device)
        
        print(f"Transcribing {audio_path}...")
        result = model.transcribe(audio_path)
        
        transcript = result["text"]
        segments = result["segments"]
        
        print(f"Transcription complete. Length: {len(transcript)} chars.")
        
        state["transcript"] = transcript
        state["segments"] = segments
        state["word_count"] = len(transcript.split())
        
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"]["transcription_model"] = "openai-whisper-base"

    except Exception as e:
        print(f"Error in A2 node: {e}")
        raise e

    if state.get("debug", False):
        print(f"[DEBUG] A2: Transcription Model: openai-whisper-base")
        print(f"[DEBUG] A2: Device: {device}")
        print(f"[DEBUG] A2: Word Count: {state.get('word_count')}")
        print(f"[DEBUG] A2: Segments: {len(state.get('segments', []))}")

    return state
