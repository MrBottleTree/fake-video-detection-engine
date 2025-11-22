import cv2
import os
import numpy as np
import urllib.request

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
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        prototxt_path = os.path.join(model_dir, "deploy.prototxt")
        model_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        
        lbfgs_path = os.path.join(model_dir, "lbfmodel.yaml")

        def download_model_file(url, path):
            if not os.path.exists(path):
                print(f"Downloading {os.path.basename(path)}...")
                try:
                    urllib.request.urlretrieve(url, path)
                    print(f"Downloaded {path}")
                except Exception as e:
                    print(f"Failed to download {url}: {e}")
                    raise e

        download_model_file(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            prototxt_path
        )
        download_model_file(
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            model_path
        )
        
        download_model_file(
            "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml",
            lbfgs_path
        )

        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            
            dummy_blob = np.zeros((1, 3, 300, 300), dtype=np.float32)
            net.setInput(dummy_blob)
            net.forward()
            
            if debug:
                print("[DEBUG] V3: CUDA backend successfully initialized.")
        except Exception as e:
            if debug:
                print(f"[DEBUG] V3: CUDA backend failed ({e}), falling back to CPU.")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        facemark = cv2.face.createFacemarkLBF()
        facemark.loadModel(lbfgs_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
            
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < 0.5:
                    continue
                
                box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
                (startX, startY, endX, endY) = box.astype("int")
                
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(frame_width, endX)
                endY = min(frame_height, endY)
                
                w = endX - startX
                h = endY - startY
                
                if w > 0 and h > 0:
                    faces.append((startX, startY, w, h))
            
            faces.sort(key=lambda f: f[2] * f[3], reverse=True)
            
            if debug and frame_count % 30 == 0:
                print(f"[DEBUG] Frame {frame_count}: Found {len(faces)} faces.")

            best_face_landmarks = []
            max_lip_distance = -1.0
            frame_landmarks = []

            for face_idx, face in enumerate(faces):
                fx, fy, fw, fh = face
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), max(2, frame_width // 300))
                
                faces_np = np.array([face], dtype=int) 
                
                success, landmarks = facemark.fit(frame, faces_np)
                
                if success:
                    shape = landmarks[0]
                    shape = np.squeeze(shape)
                    
                    if shape.shape[0] >= 68:
                        mouth_points = shape[48:68]
                        
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
                            
                            cv2.putText(frame, f"D:{distance:.1f}", (fx, max(0, fy - 10)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), line_thickness)
                            
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
        state["metadata"]["landmark_model"] = "opencv_lbfgs"

    except Exception as e:
        print(f"Error in V3 node: {e}")
        raise e

    return state
