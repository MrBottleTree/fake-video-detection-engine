import cv2
import numpy as np
import os
from PIL import Image, ImageChops, ImageEnhance
import base64
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

def run(state: dict) -> dict:
    print("Node V5: Running Texture & ELA Analysis...")
    
    face_detections = state.get("face_detections", [])
    debug = state.get("debug", False)
    output_dir = state.get("data_dir")
    
    if not face_detections:
        print("Node V5: No faces detected to analyze.")
        state["texture_ela_score"] = 0.0
        state["texture_ela_details"] = {"reason": "No faces found"}
        return state

    # 1. Select representative faces (up to 3)
    # Strategy: Pick faces with high confidence and large area from different parts of the video
    sorted_faces = sorted(face_detections, key=lambda x: (x['faces'][0]['confidence'] * x['faces'][0]['bbox']['w'] * x['faces'][0]['bbox']['h']), reverse=True)
    
    # Simple selection: Top 3 best faces
    selected_faces = sorted_faces[:3]
    
    ela_dir = os.path.join(output_dir, "ela_analysis")
    os.makedirs(ela_dir, exist_ok=True)
    
    analysis_results = []
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    for i, face_data in enumerate(selected_faces):
        try:
            face_info = face_data['faces'][0]
            crop_path = face_info['crop_path']
            
            if not os.path.exists(crop_path):
                continue
                
            # --- ELA Analysis ---
            original = Image.open(crop_path).convert('RGB')
            
            # Save at 90% quality to a temporary buffer/file
            ela_temp_path = os.path.join(ela_dir, f"temp_ela_{i}.jpg")
            original.save(ela_temp_path, 'JPEG', quality=90)
            compressed = Image.open(ela_temp_path)
            
            # Calculate difference
            diff = ImageChops.difference(original, compressed)
            
            # Enhance brightness to make artifacts visible
            extrema = diff.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            if max_diff == 0:
                max_diff = 1
            scale = 255.0 / max_diff
            
            enhanced_diff = ImageEnhance.Brightness(diff).enhance(scale)
            
            ela_output_path = os.path.join(ela_dir, f"ela_{i}.jpg")
            enhanced_diff.save(ela_output_path)
            
            # --- FFT Analysis ---
            # Convert to grayscale for FFT
            gray_image = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
            f = np.fft.fft2(gray_image)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
            
            # Normalize for visualization
            magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            fft_output_path = os.path.join(ela_dir, f"fft_{i}.jpg")
            cv2.imwrite(fft_output_path, magnitude_spectrum)
            
            # --- OpenAI Analysis ---
            if client.api_key:
                def encode_image(image_path):
                    with open(image_path, "rb") as image_file:
                        return base64.b64encode(image_file.read()).decode('utf-8')

                base64_original = encode_image(crop_path)
                base64_ela = encode_image(ela_output_path)
                base64_fft = encode_image(fft_output_path)
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a forensic image analyst specializing in deepfake detection. Analyze the provided images: 1. Original Face, 2. Error Level Analysis (ELA) map, 3. FFT Magnitude Spectrum. Look for: blending artifacts in ELA (high contrast edges around eyes/mouth), inconsistent noise patterns, or grid-like artifacts in FFT (GAN fingerprints). Provide a 'fake_probability' score (0.0 to 1.0) and a brief reasoning."
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Analyze this face for manipulation."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_original}"}},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_ela}"}},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_fft}"}}
                            ]
                        }
                    ],
                    response_format={"type": "json_object"}
                )
                
                result_json = json.loads(response.choices[0].message.content)
                analysis_results.append(result_json)
                
        except Exception as e:
            print(f"Error analyzing face {i}: {e}")
            if debug:
                import traceback
                traceback.print_exc()

    # Aggregate scores
    if analysis_results:
        avg_score = sum(r.get('fake_probability', 0) for r in analysis_results) / len(analysis_results)
        state["texture_ela_score"] = avg_score
        state["texture_ela_details"] = analysis_results
        print(f"Node V5: Analysis complete. Score: {avg_score:.2f}")
    else:
        print("Node V5: No analysis results generated.")
        state["texture_ela_score"] = 0.0
        state["texture_ela_details"] = {"reason": "Analysis failed or no keys"}

    return state
