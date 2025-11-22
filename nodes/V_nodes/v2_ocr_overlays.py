import easyocr
import os

def run(state: dict) -> dict:
    print("Node V2: Extracting text from keyframes (OCR)...")
    keyframes = state.get("keyframes", [])
    debug = state.get("debug", False)
    if debug:
        print("Keyframes:", keyframes)
    
    if not keyframes:
        print("Warning: No keyframes found in state. Skipping OCR.")
        return state

    try:
        use_gpu = False
        try:
            import torch
            if torch.cuda.is_available():
                use_gpu = True
                if debug:
                    print("[DEBUG] V2: CUDA available, using GPU for EasyOCR.")
            else:
                if debug:
                    print("[DEBUG] V2: CUDA not available, using CPU for EasyOCR.")
        except ImportError:
            if debug:
                print("[DEBUG] V2: torch not found, using CPU for EasyOCR.")

        reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=debug)
        
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
