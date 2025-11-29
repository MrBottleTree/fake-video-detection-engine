import os
import json
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import base64
from io import BytesIO
from nodes import dump_node_debug

def encode_image_base64(image_path):
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def find_closest_segment(timestamp: float, segments: list, tolerance: float = 2.0) -> dict:
    """Finds the segment closest to the timestamp within a tolerance window."""
    best_seg = None
    min_dist = float('inf')
    
    for seg in segments:
        start, end = seg["start"], seg["end"]
        # Check strict overlap first
        if start <= timestamp <= end:
            return seg
        
        # Calculate distance to segment boundaries
        dist = min(abs(timestamp - start), abs(timestamp - end))
        if dist < min_dist and dist <= tolerance:
            min_dist = dist
            best_seg = seg
            
    return best_seg

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
        # Extract timestamp from filename
        timestamp = None
        frame_id = -1
        try:
            basename = os.path.basename(kf_path)
            # Robust parsing: handle various filename formats
            # Expected: frame_{id}.jpg or keyframe_{id}.jpg
            parts = basename.replace('.', '_').split('_')
            for p in parts:
                if p.isdigit():
                    frame_id = int(p)
                    break
            
            if frame_id != -1:
                timestamp = frame_id / fps
            else:
                if debug:
                    print(f"[DEBUG] Could not parse frame ID from {basename}")
                continue
        except Exception as e:
            if debug:
                print(f"[DEBUG] Error parsing timestamp from {kf_path}: {e}")
            continue

        # Find corresponding segment (Fuzzy Match)
        matched_seg = find_closest_segment(timestamp, segments, tolerance=2.0)
        
        if not matched_seg:
            if debug:
                print(f"[DEBUG] No segment found for frame {frame_id} (t={timestamp:.2f}s)")
            continue
            
        current_text = matched_seg["text"]

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
                        content = response.choices[0].message.content
                        if not content:
                            raise ValueError("OpenAI returned empty content")
                            
                        result = json.loads(content)
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
    dump_node_debug(
        state,
        "C2",
        {
            "checked": len(gesture_checks),
            "matched": sum(1 for g in gesture_checks if g.get("status") == "Consistent"),
            "inconsistent": sum(1 for g in gesture_checks if g.get("status") == "Inconsistent"),
        },
    )
    
    print("C2 returning")
    return state
