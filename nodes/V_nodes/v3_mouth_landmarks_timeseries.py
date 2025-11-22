import cv2
import os
import numpy as np
import face_alignment
import torch

def run(state: dict) -> dict:
    print("Node V3: Extracting mouth landmarks (Time-series) with Face Alignment...")
    output_dir = state.get("data_dir")
    debug = state.get("debug", False)
    
    if not output_dir or not os.path.exists(output_dir):
        print(f"Error: Data directory not found at {output_dir}")
        return state

    video_path = os.path.join(output_dir, "video.mp4")
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return state

    try:
        # Initialize Face Alignment
        # 2D alignment, using CUDA if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[DEBUG] V3: Using device: {device}")
        
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device, face_detector='sfd')

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        viz_path = os.path.join(output_dir, "landmarks_viz.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        viz_writer = cv2.VideoWriter(viz_path, fourcc, fps, (frame_width, frame_height))
        
        mouth_landmarks_data = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Face Alignment expects RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect landmarks
            # get_landmarks returns a list of numpy arrays, one for each face
            # Each array is (68, 2)
            try:
                landmarks_list = fa.get_landmarks(frame_rgb)
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Frame {frame_count}: Error in landmark detection: {e}")
                landmarks_list = None

            best_face_landmarks = []
            max_lip_distance = -1.0
            frame_landmarks = []

            if landmarks_list:
                for landmarks in landmarks_list:
                    # landmarks is (68, 2)
                    
                    # Calculate bounding box for visualization
                    x_min = int(np.min(landmarks[:, 0]))
                    y_min = int(np.min(landmarks[:, 1]))
                    x_max = int(np.max(landmarks[:, 0]))
                    y_max = int(np.max(landmarks[:, 1]))
                    
                    w = x_max - x_min
                    h = y_max - y_min
                    
                    # Draw bounding box (Blue)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), max(2, frame_width // 300))

                    # Extract mouth points (48-68)
                    mouth_points = landmarks[48:68]
                    
                    # Dynamic Scaling
                    circle_radius = max(3, frame_width // 200)
                    line_thickness = max(2, frame_width // 300)
                    font_scale = max(1.0, frame_width / 1000.0)
                    
                    # Draw landmarks (Green)
                    for (x, y) in mouth_points:
                        cv2.circle(frame, (int(x), int(y)), circle_radius, (0, 255, 0), -1)
                    
                    # Calculate lip distance
                    # 62 is top inner lip, 66 is bottom inner lip (0-indexed from 0-67)
                    # In 48-67 subset: 62->14, 66->18
                    if len(mouth_points) >= 20:
                        top_lip = mouth_points[14]
                        bottom_lip = mouth_points[18]
                        
                        distance = np.linalg.norm(top_lip - bottom_lip)
                        
                        # Draw line (Red)
                        cv2.line(frame, (int(top_lip[0]), int(top_lip[1])), 
                                 (int(bottom_lip[0]), int(bottom_lip[1])), (0, 0, 255), line_thickness)
                        
                        # Display distance
                        cv2.putText(frame, f"D:{distance:.1f}", (x_min, max(0, y_min - 10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), line_thickness)
                        
                        # Check if this is the "best" face (widest mouth)
                        if distance > max_lip_distance:
                            max_lip_distance = distance
                            best_face_landmarks = mouth_points.tolist()

            if debug and frame_count < 5 and len(best_face_landmarks) > 0:
                debug_img_path = os.path.join(output_dir, f"debug_v3_frame_{frame_count}.jpg")
                cv2.imwrite(debug_img_path, frame)
                print(f"[DEBUG] Saved debug image to {debug_img_path}")
            
            frame_landmarks = best_face_landmarks
            
            viz_writer.write(frame)
            
            mouth_landmarks_data.append({
                "frame_id": frame_count,
                "timestamp": frame_count / fps,
                "landmarks": frame_landmarks
            })
            
            frame_count += 1
            
            if debug and frame_count % 30 == 0:
                print(f"[DEBUG] V3: Processed {frame_count} frames...")

        cap.release()
        viz_writer.release()
        
        print(f"Extracted mouth landmarks for {len(mouth_landmarks_data)} frames.")
        print(f"Visualization saved to {viz_path}")
        
        state["mouth_landmarks"] = mouth_landmarks_data
        state["mouth_landmarks_viz_path"] = viz_path
        
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"]["landmark_model"] = "face_alignment_sfd"

    except Exception as e:
        print(f"Error in V3 node: {e}")
        raise e

    return state
