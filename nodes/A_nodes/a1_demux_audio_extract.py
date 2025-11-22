import os
from moviepy import AudioFileClip

def run(state: dict) -> dict:
    print("Node A1: Standardizing audio...")
    output_dir = state.get("data_dir")
    
    if not output_dir or not os.path.exists(output_dir):
        print(f"Error: Data directory not found at {output_dir}")
        return state

    try:
        output_filename = "audio_16k.wav"
        output_path = os.path.join(output_dir, output_filename)

        audio_input_path = os.path.join(output_dir, "audio.wav")
        if not os.path.exists(audio_input_path):
            print(f"Warning: Audio file not found at {audio_input_path}. Skipping audio standardization.")
            return state

        clip = AudioFileClip(audio_input_path)
        
        clip.write_audiofile(
            output_path, 
            fps=16000, 
            nbytes=2, 
            codec='pcm_s16le', 
            ffmpeg_params=["-ac", "1"],
            logger=None
        )
        
        clip.close()

        print(f"Audio standardized to: {output_path}")
        
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"]["audio_sample_rate"] = 16000
        state["metadata"]["audio_channels"] = 1

    except Exception as e:
        print(f"Error standardizing audio: {e}")
        raise e

    if state.get("debug", False):
        print(f"[DEBUG] A1: Audio standardized to {output_path}")
        print(f"[DEBUG] A1: Sample Rate: 16000, Channels: 1")

    return state
