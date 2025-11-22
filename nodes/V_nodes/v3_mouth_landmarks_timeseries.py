import cv2
import os
import numpy as np
import mediapipe as mp

def run(state: dict) -> dict:
    print("Node V3: Extracting mouth landmarks (Time-series)...")
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
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        if debug:
            print("[DEBUG] V3: MediaPipe Face Mesh initialized with GPU acceleration")
            print("[DEBUG] V3: Refine landmarks enabled for enhanced lip tracking")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if debug:
            print(f"[DEBUG] V3: Video FPS: {fps}, Size: {frame_width}x{frame_height}, Frames: {total_frames}")
        
        viz_path = os.path.join(output_dir, "landmarks_viz.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        viz_writer = cv2.VideoWriter(viz_path, fourcc, fps, (frame_width, frame_height))
        
        mouth_landmarks_data = []
        frame_count = 0
        
        UPPER_LIP_CENTER = 13
        LOWER_LIP_CENTER = 14
        
        MOUTH_LANDMARKS = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95
        ]
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = face_mesh.process(rgb_frame)
            
            best_face_landmarks = []
            max_lip_distance = -1.0
            
            if results.multi_face_landmarks:
                if debug and frame_count % 30 == 0:
                    print(f"[DEBUG] V3: Frame {frame_count}: Found {len(results.multi_face_landmarks)} faces")
                
                for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_LIPS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    
                    mouth_points = []
                    for idx in MOUTH_LANDMARKS:
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * frame_width)
                        y = int(landmark.y * frame_height)
                        mouth_points.append([x, y])
                        
                        cv2.circle(frame, (x, y), max(2, frame_width // 400), (0, 255, 0), -1)
                    
                    upper_lip = face_landmarks.landmark[UPPER_LIP_CENTER]
                    lower_lip = face_landmarks.landmark[LOWER_LIP_CENTER]
                    
                    upper_lip_px = np.array([upper_lip.x * frame_width, upper_lip.y * frame_height])
                    lower_lip_px = np.array([lower_lip.x * frame_width, lower_lip.y * frame_height])
                    
                    distance = np.linalg.norm(upper_lip_px - lower_lip_px)
                    
                    line_thickness = max(2, frame_width // 300)
                    cv2.line(frame, 
                            (int(upper_lip_px[0]), int(upper_lip_px[1])),
                            (int(lower_lip_px[0]), int(lower_lip_px[1])),
                            (0, 0, 255), line_thickness)
                    
                    font_scale = max(0.5, frame_width / 1000.0)
                    cv2.putText(frame, f"D:{distance:.1f}", 
                               (int(upper_lip_px[0]), int(upper_lip_px[1]) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), line_thickness)
                    
                    if distance > max_lip_distance:
                        max_lip_distance = distance
                        best_face_landmarks = mouth_points
            
            if debug and frame_count < 5 and len(best_face_landmarks) > 0:
                debug_img_path = os.path.join(output_dir, f"debug_v3_frame_{frame_count}.jpg")
                cv2.imwrite(debug_img_path, frame)
                print(f"[DEBUG] V3: Saved debug image to {debug_img_path}")
                print(f"[DEBUG] V3: Detected {len(best_face_landmarks)} mouth landmarks")
            
            viz_writer.write(frame)
            
            mouth_landmarks_data.append({
                "frame_id": frame_count,
                "timestamp": frame_count / fps,
                "landmarks": best_face_landmarks,
                "lip_distance": max_lip_distance if max_lip_distance > 0 else 0.0
            })
            
            frame_count += 1
            
            if debug and frame_count % 30 == 0:
                print(f"[DEBUG] V3: Processed {frame_count}/{total_frames} frames...")

        cap.release()
        viz_writer.release()
        face_mesh.close()
        
        print(f"Extracted mouth landmarks for {len(mouth_landmarks_data)} frames.")
        print(f"Visualization saved to {viz_path}")
        
        state["mouth_landmarks"] = mouth_landmarks_data
        state["mouth_landmarks_viz_path"] = viz_path
        
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"]["landmark_model"] = "mediapipe_face_mesh"
        state["metadata"]["landmark_count"] = 40

    except Exception as e:
        print(f"Error in V3 node: {e}")
        raise e

    return state
