import cv2
import os
import numpy as np
import face_alignment
import torch
from scipy.spatial import distance as dist
import math
from sixdrepnet import SixDRepNet

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = float(x0) if isinstance(x0, (float, int)) else np.array(x0, dtype=float)
        self.dx_prev = float(dx0) if isinstance(dx0, (float, int)) else np.array(dx0, dtype=float)
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    # Referenced from SixDRepNet utils
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (math.cos(yaw) * math.cos(roll)) + tdx
    y1 = size * (math.cos(pitch) * math.sin(roll) + math.cos(roll) * math.sin(pitch) * math.sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-math.cos(yaw) * math.sin(roll)) + tdx
    y2 = size * (math.cos(pitch) * math.cos(roll) - math.sin(pitch) * math.sin(yaw) * math.sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (math.sin(yaw)) + tdx
    y3 = size * (-math.cos(yaw) * math.sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def run(state: dict) -> dict:
    print("Node V4: Analyzing Blink (FaceAlignment) and Head Pose (SixDRepNet)...")
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
        # Initialize Face Alignment in 3D mode (for landmarks/blink)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if debug:
            print(f"[DEBUG] V4: Using device: {device}")
            
        # Suppress "No faces were detected" warning
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="face_alignment")
            
        # 3D Landmarks for better accuracy
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, device=device, face_detector='sfd', face_detector_kwargs={'filter_threshold': 0.5})

        # Initialize SixDRepNet for Robust Head Pose
        # dict_path=None will download weights automatically if not found
        pose_model = SixDRepNet(gpu_id=0 if device == 'cuda' else -1)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        blink_data = []
        head_pose_data = []
        frame_count = 0
        
        active_face_box = None
        landmark_filter = None
        pose_filter = None # OneEuroFilter for pose angles
        
        viz_path = os.path.join(output_dir, "headpose_viz.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        viz_writer = cv2.VideoWriter(viz_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_time = frame_count / fps
            
            try:
                landmarks_list = fa.get_landmarks_from_image(frame_rgb)
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Frame {frame_count}: Error in landmark detection: {e}")
                landmarks_list = None
            
            current_ear = None
            current_pose = None
            
            if landmarks_list:
                best_face_idx = -1
                
                # Tracking Logic
                if active_face_box is None:
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
                else:
                    max_iou = -1
                    for i, landmarks in enumerate(landmarks_list):
                        x_min = int(np.min(landmarks[:, 0]))
                        y_min = int(np.min(landmarks[:, 1]))
                        x_max = int(np.max(landmarks[:, 0]))
                        y_max = int(np.max(landmarks[:, 1]))
                        
                        current_box = [x_min, y_min, x_max, y_max]
                        iou = calculate_iou(active_face_box, current_box)
                        
                        if iou > max_iou:
                            max_iou = iou
                            best_face_idx = i
                    
                    if max_iou < 0.3:
                        if debug:
                            print(f"[DEBUG] Frame {frame_count}: Tracking lost (IoU {max_iou:.2f}), resetting.")
                        active_face_box = None
                        # Fallback to largest face
                        max_area = -1
                        for i, landmarks in enumerate(landmarks_list):
                            x_min = int(np.min(landmarks[:, 0]))
                            y_min = int(np.min(landmarks[:, 1]))
                            x_max = int(np.max(landmarks[:, 0]))
                            y_max = int(np.max(landmarks[:, 1]))
                            area = (x_max - x_min) * (y_max - y_min)
                            if area > max_area:
                                max_area = area
                                best_face_idx = i

                if best_face_idx != -1:
                    raw_landmarks = landmarks_list[best_face_idx]
                    
                    x_min = int(np.min(raw_landmarks[:, 0]))
                    y_min = int(np.min(raw_landmarks[:, 1]))
                    x_max = int(np.max(raw_landmarks[:, 0]))
                    y_max = int(np.max(raw_landmarks[:, 1]))
                    
                    # Expand box slightly for SixDRepNet
                    pad_w = int((x_max - x_min) * 0.1)
                    pad_h = int((y_max - y_min) * 0.1)
                    x_min = max(0, x_min - pad_w)
                    y_min = max(0, y_min - pad_h)
                    x_max = min(frame_width, x_max + pad_w)
                    y_max = min(frame_height, y_max + pad_h)
                    
                    active_face_box = [x_min, y_min, x_max, y_max]
                    
                    # 1. Blink Detection (EAR) - Smoothed
                    if landmark_filter is None:
                        landmark_filter = OneEuroFilter(current_time, raw_landmarks, min_cutoff=0.5, beta=0.1)
                        smoothed_landmarks = raw_landmarks
                    else:
                        smoothed_landmarks = landmark_filter(current_time, raw_landmarks)
                        
                    leftEye = smoothed_landmarks[36:42]
                    rightEye = smoothed_landmarks[42:48]
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0
                    current_ear = ear
                    
                    # 2. Robust Head Pose (SixDRepNet)
                    # Crop face
                    face_img = frame[y_min:y_max, x_min:x_max]
                    if face_img.size > 0:
                        # Predict
                        pitch, yaw, roll = pose_model.predict(face_img)
                        # SixDRepNet returns single values usually, ensure float
                        pitch = float(pitch[0])
                        yaw = float(yaw[0])
                        roll = float(roll[0])
                        
                        # Smooth Pose
                        pose_vec = np.array([pitch, yaw, roll])
                        if pose_filter is None:
                            pose_filter = OneEuroFilter(current_time, pose_vec, min_cutoff=0.1, beta=0.1) # Stronger smoothing for pose
                            smoothed_pose = pose_vec
                        else:
                            smoothed_pose = pose_filter(current_time, pose_vec)
                            
                        pitch, yaw, roll = smoothed_pose
                        current_pose = {"pitch": pitch, "yaw": yaw, "roll": roll}
                        
                        # Visualization
                        # Draw Axis at nose center (landmark 30)
                        nose_x = int(smoothed_landmarks[30][0])
                        nose_y = int(smoothed_landmarks[30][1])
                        draw_axis(frame, yaw, pitch, roll, tdx=nose_x, tdy=nose_y, size=50)

                    # Draw Eyes
                    for (x, y, z) in np.concatenate((leftEye, rightEye), axis=0):
                        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                    
                    # Draw Box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 1)
                    
                    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if current_pose:
                        cv2.putText(frame, f"Y:{yaw:.0f} P:{pitch:.0f} R:{roll:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "SixDRepNet+Smooth", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            viz_writer.write(frame)
            
            if current_ear is not None:
                blink_data.append({
                    "frame_id": frame_count,
                    "timestamp": current_time,
                    "ear": current_ear
                })
            
            if current_pose is not None:
                head_pose_data.append({
                    "frame_id": frame_count,
                    "timestamp": current_time,
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
        state["metadata"]["blink_model"] = "EAR_threshold_3D_smoothed"
        state["metadata"]["pose_model"] = "SixDRepNet_smoothed"

    except Exception as e:
        print(f"Error in V4 node: {e}")
        raise e

    return state
