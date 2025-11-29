import os
import whisper
import torch
import imageio_ffmpeg
import warnings
from nodes import dump_node_debug

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

def run(state: dict) -> dict:
    print("Node A2: Running VAD and ASR...")
    
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_path)
    if ffmpeg_dir not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + ffmpeg_dir

    audio_path = os.path.join(state.get("data_dir"), "audio_16k.wav")
    
    if not audio_path or not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return state

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Node A2: Whisper running on {device.upper()}")
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
        dump_node_debug(
            state,
            "A2",
            {
                "words": state.get("word_count", 0),
                "segments": len(state.get("segments", [])),
                "device": device,
            },
        )

    except Exception as e:
        print(f"Error in A2 node: {e}")
        raise e

    if state.get("debug", False):
        print(f"[DEBUG] A2: Transcription Model: openai-whisper-base")
        print(f"[DEBUG] A2: Device: {device}")
        print(f"[DEBUG] A2: Word Count: {state.get('word_count')}")
        print(f"[DEBUG] A2: Segments: {len(state.get('segments', []))}")

    print("Node A2 returning state...", flush=True)
    return state
