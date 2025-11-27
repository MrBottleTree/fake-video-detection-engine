import os
import json
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import base64
from io import BytesIO

def encode_image_base64(image_path):
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def run(state: dict) -> dict:
    print("Node C2: Checking Gesture-Narration Consistency (CLIP + OpenAI)...")
    
    keyframes = state.get("keyframes", [])
    transcript = state.get("transcript", "")
    segments = state.get("segments", [])
    debug = state.get("debug", False)
    
    if not keyframes:
        print("Warning: No keyframes found. Skipping gesture check.")
        return state

    # Load CLIP model (Local GPU/CPU)
    try:
        model_name = "clip-ViT-B-32"
        if debug:
            print(f"[DEBUG] C2: Loading CLIP model {model_name}...")
        model = SentenceTransformer(model_name)
        if debug:
            print(f"[DEBUG] C2: Model loaded.")
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        return state

    # Initialize OpenAI client (lazy load only if needed? No, init here to check key)
    openai_client = None
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        openai_client = OpenAI(api_key=api_key)
    else:
        print("Warning: OPENAI_API_KEY not found. Fallback will be disabled.")

    gesture_checks = []
    
    # Get FPS for timestamp calculation if needed, though keyframes usually have paths like frame_XXXXXX.jpg
    # We need to map keyframe to a text segment.
    # Assuming keyframe filename format: frame_{frame_id:06d}.jpg
    # And we need FPS to convert frame_id to time.
    fps = state.get("metadata", {}).get("video_fps", 30.0)
    
    for kf_path in keyframes:
        if not os.path.exists(kf_path):
            continue
            
        # Extract timestamp from filename
        try:
            basename = os.path.basename(kf_path)
            frame_id = int(basename.split('_')[1].split('.')[0])
            timestamp = frame_id / fps
        except:
            if debug:
                print(f"[DEBUG] Could not parse timestamp from {kf_path}")
            continue

        # Find corresponding segment
        current_text = ""
        for seg in segments:
            if seg["start"] <= timestamp <= seg["end"]:
                current_text = seg["text"]
                break
        
        if not current_text:
            # If no exact segment, maybe take the closest one or skip
            continue

        # 1. CLIP Check
        try:
            img_emb = model.encode(Image.open(kf_path))
            text_emb = model.encode(current_text)
            
            # Cosine similarity
            score = util.cos_sim(img_emb, text_emb).item()
            
            status = "Uncertain"
            reason = f"CLIP Score: {score:.2f}"
            source = "clip_local"
            
            # Thresholds
            if score > 0.25:
                status = "Consistent"
            elif score < 0.15:
                status = "Inconsistent"
            else:
                # Ambiguous -> OpenAI Fallback
                if openai_client:
                    if debug:
                        print(f"[DEBUG] C2: Ambiguous score ({score:.2f}) for frame {frame_id}. Triggering OpenAI fallback.")
                    
                    try:
                        base64_image = encode_image_base64(kf_path)
                        response = openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": f"Does the image support this narration: '{current_text}'? Return JSON with 'consistent' (bool) and 'reason'."},
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpeg;base64,{base64_image}"
                                            }
                                        }
                                    ]
                                }
                            ],
                            response_format={"type": "json_object"}
                        )
                        result = json.loads(response.choices[0].message.content)
                        status = "Consistent" if result.get("consistent") else "Inconsistent"
                        reason = f"OpenAI Fallback: {result.get('reason')}"
                        source = "openai_fallback"
                        
                    except Exception as e:
                        print(f"OpenAI Fallback failed: {e}")
                        status = "Ambiguous" # Keep as ambiguous if fallback fails
                else:
                    status = "Ambiguous (No API Key)"

            gesture_checks.append({
                "timestamp": timestamp,
                "frame_id": frame_id,
                "text": current_text,
                "status": status,
                "score": score,
                "reason": reason,
                "source": source
            })
            
        except Exception as e:
            if debug:
                print(f"[DEBUG] Error processing frame {kf_path}: {e}")
            continue

    print(f"Node C2: Checked {len(gesture_checks)} frames.")
    state["gesture_check"] = gesture_checks
    
    return state
