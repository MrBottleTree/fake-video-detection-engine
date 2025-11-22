import cv2
import os
import numpy as np
import face_alignment
import torch
from scipy.spatial import distance as dist

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
 
    return ear

def get_head_pose(shape, frame_width, frame_height):
    image_points = np.array([
                            shape[30],
                            shape[8],
                            shape[36],
                            shape[45],
                            shape[48],
                            shape[54]
                        ], dtype="double")
 
    model_points = np.array([
                            (0.0, 0.0, 0.0),
                            (0.0, -330.0, -65.0),
                            (-225.0, 170.0, -135.0),
                            (225.0, 170.0, -135.0),
                            (-150.0, -150.0, -125.0),
                            (150.0, -150.0, -125.0)
                        ])
 
    focal_length = frame_width
    center = (frame_width/2, frame_height/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )
 
    dist_coeffs = np.zeros((4,1))
    
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    
    rmat, jac = cv2.Rodrigues(rotation_vector)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
    pitch = angles[0]
    yaw = angles[1]
    roll = angles[2]
    
    return pitch, yaw, roll, nose_end_point2D

def run(state: dict) -> dict:
    print("Node V4: Analyzing Blink and Head Pose Dynamics...")
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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if debug:
            print(f"[DEBUG] V4: Using device: {device}")
            
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device, face_detector='sfd')

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        blink_data = []
        head_pose_data = []
        frame_count = 0
        
        viz_path = os.path.join(output_dir, "headpose_viz.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        viz_writer = cv2.VideoWriter(viz_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                landmarks_list = fa.get_landmarks(frame_rgb)
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Frame {frame_count}: Error in landmark detection: {e}")
                landmarks_list = None
            
            current_ear = None
            current_pose = None
            
            if landmarks_list:
                best_face_idx = -1
                max_area = -1
                
                for i, landmarks in enumerate(landmarks_list):
                    x_min = int(np.min(landmarks[:, 0]))
                    y_min = int(np.min(landmarks[:, 1]))
                    x_max = int(np.max(landmarks[:, 0]))
                    y_max = int(np.max(landmarks[:, 1]))
                    
                    w = x_max - x_min
                    h = y_max - y_min
                    
                    if w < (frame_width * 0.05) or h < (frame_height * 0.05):
                        continue
                    
                    area = w * h
                    if area > max_area:
                        max_area = area
                        best_face_idx = i
                
                if best_face_idx != -1:
                    landmarks = landmarks_list[best_face_idx]
                    
                    leftEye = landmarks[36:42]
                    rightEye = landmarks[42:48]
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0
                    current_ear = ear
                    
                    pitch, yaw, roll, nose_end_point2D = get_head_pose(landmarks, frame_width, frame_height)
                    current_pose = {"pitch": pitch, "yaw": yaw, "roll": roll}
                    
                    for (x, y) in np.concatenate((leftEye, rightEye), axis=0):
                        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                        
                    p1 = (int(landmarks[30][0]), int(landmarks[30][1]))
                    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                    cv2.line(frame, p1, p2, (255, 0, 0), 2)
                    
                    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f"Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            viz_writer.write(frame)
            
            if current_ear is not None:
                blink_data.append({
                    "frame_id": frame_count,
                    "timestamp": frame_count / fps,
                    "ear": current_ear
                })
            
            if current_pose is not None:
                head_pose_data.append({
                    "frame_id": frame_count,
                    "timestamp": frame_count / fps,
                    "pose": current_pose
                })
            
            frame_count += 1
            if debug and frame_count % 50 == 0:
                 print(f"[DEBUG] V4: Processed {frame_count} frames...")

        cap.release()
        viz_writer.release()
        
        print(f"V4 Analysis complete. Extracted {len(blink_data)} blink samples and {len(head_pose_data)} pose samples.")
        
        state["blink_data"] = blink_data
        state["head_pose_data"] = head_pose_data
        state["headpose_viz_path"] = viz_path
        
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"]["blink_model"] = "EAR_threshold"
        state["metadata"]["pose_model"] = "SolvePnP"

    except Exception as e:
        print(f"Error in V4 node: {e}")
        raise e

    return state
