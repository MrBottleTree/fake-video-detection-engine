import easyocr
import os

def run(state: dict) -> dict:
    print("Node V2: Extracting text from keyframes (OCR)...")
    keyframes = state.get("keyframes", [])
    print("Keyframes:", keyframes)
    debug = state.get("debug", False)
    
    if not keyframes:
        print("Warning: No keyframes found in state. Skipping OCR.")
        return state

    try:
        reader = easyocr.Reader(['en'], gpu=False, verbose=debug)
        
        ocr_results = []
        
        for kf_path in keyframes:
            if not os.path.exists(kf_path):
                if debug:
                    print(f"[DEBUG] V2: Keyframe not found: {kf_path}")
                continue
                
            if debug:
                print(f"[DEBUG] V2: Processing {os.path.basename(kf_path)}...")
            
            results = reader.readtext(kf_path, detail=1)
            
            frame_text = []
            for (bbox, text, prob) in results:
                clean_bbox = [[int(p[0]), int(p[1])] for p in bbox]
                
                frame_text.append({
                    "text": text,
                    "confidence": float(prob),
                    "bbox": clean_bbox
                })
            
            if frame_text:
                ocr_results.append({
                    "keyframe_path": kf_path,
                    "detections": frame_text
                })
                
        print(f"OCR complete. Found text in {len(ocr_results)} frames.")
        
        state["ocr_results"] = ocr_results
        
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"]["ocr_model"] = "easyocr_en"

    except Exception as e:
        print(f"Error in V2 node: {e}")
        raise e

    return state
