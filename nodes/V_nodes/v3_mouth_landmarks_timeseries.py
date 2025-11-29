import cv2
import os
import numpy as np
import face_alignment
import torch
import math
from nodes import dump_node_debug

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
    print("DEBUG: Entering Node V3 run function...")
    print("Node V3: Extracting mouth landmarks (Time-series) with Robust Tracking...")
    output_dir = state.get("data_dir")
    
    if output_dir and os.path.exists(output_dir):
        with open(os.path.join(output_dir, "debug_log.txt"), "a") as f:
            f.write("Node V3 started.\n")
            
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
        print(f"Node V3: Face Alignment running on {device.upper()}")
        if debug:
            print(f"[DEBUG] V3: Using device: {device}")
        
        # Suppress "No faces were detected" warning
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="face_alignment")
        
        # Lower threshold for better recall
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device, face_detector='sfd', face_detector_kwargs={'filter_threshold': 0.5})

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        target_fps = 5.0  # sample for speed; full frame-by-frame is too slow
        sample_stride = max(1, int(round(fps / target_fps))) if fps else 1
        viz_fps = max(1.0, fps / sample_stride) if fps else 1.0
        
        viz_path = os.path.join(output_dir, "landmarks_viz.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        viz_writer = cv2.VideoWriter(viz_path, fourcc, viz_fps, (frame_width, frame_height))
        
        mouth_landmarks_data = []
        frame_idx = 0
        processed_frames = 0
        
        active_face_box = None
        landmark_filter = None
        
        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_time = frame_idx / fps if fps else processed_frames
            
            try:
                landmarks_list = fa.get_landmarks(frame_rgb)
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Frame {frame_idx}: Error in landmark detection: {e}")
                landmarks_list = None

            best_face_idx = -1
            frame_landmarks = []

            if landmarks_list:
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
                        
                        # Aspect ratio check
                        aspect_ratio = w / h
                        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
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
                    
                    if max_iou < 0.15:
                        if debug:
                            print(f"[DEBUG] Frame {frame_idx}: Tracking lost (IoU {max_iou:.2f}), resetting.")
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
                active_face_box = [x_min, y_min, x_max, y_max]
                
                # Smoothing (OneEuroFilter)
                if landmark_filter is None:
                    landmark_filter = OneEuroFilter(current_time, raw_landmarks, min_cutoff=0.5, beta=0.1)
                    smoothed_landmarks = raw_landmarks
                else:
                    smoothed_landmarks = landmark_filter(current_time, raw_landmarks)
                
                # Visualization & Extraction
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), max(2, frame_width // 300))
                mouth_points = smoothed_landmarks[48:68]
                circle_radius = max(3, frame_width // 200)
                line_thickness = max(2, frame_width // 300)
                font_scale = max(1.0, frame_width / 1000.0)
                
                for (x, y) in mouth_points:
                    cv2.circle(frame, (int(x), int(y)), circle_radius, (0, 255, 0), -1)
                
                if len(mouth_points) >= 20:
                    top_lip = mouth_points[14]
                    bottom_lip = mouth_points[18]
                    
                    distance = np.linalg.norm(top_lip - bottom_lip)
                    
                    cv2.line(frame, (int(top_lip[0]), int(top_lip[1])), 
                                (int(bottom_lip[0]), int(bottom_lip[1])), (0, 0, 255), line_thickness)
                    
                    cv2.putText(frame, f"D:{distance:.1f}", (x_min, max(0, y_min - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), line_thickness)
                    
                    frame_landmarks = mouth_points.tolist()
                    cv2.putText(frame, "Smoothed", (x_min, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            viz_writer.write(frame)
            
            mouth_landmarks_data.append({
                "frame_id": frame_idx,
                "timestamp": current_time,
                "landmarks": frame_landmarks
            })
            
            processed_frames += 1
            frame_idx += sample_stride
            
            if debug and processed_frames % 10 == 0:
                print(f"[DEBUG] V3: Processed {processed_frames} frames (stride={sample_stride})...")

        cap.release()
        viz_writer.release()
        
        print(f"Extracted mouth landmarks for {len(mouth_landmarks_data)} frames.")
        print(f"Visualization saved to {viz_path}")
        
        state["mouth_landmarks"] = mouth_landmarks_data
        state["mouth_landmarks_viz_path"] = viz_path
        
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"]["landmark_model"] = "face_alignment_sfd_smoothed"
        dump_node_debug(
            state,
            "V3",
            {
                "frames": len(mouth_landmarks_data),
                "viz_path": viz_path,
                "fps_sampled": viz_fps,
            },
        )

    except Exception as e:
        print(f"Error in V3 node: {e}")
        raise e

    return state
