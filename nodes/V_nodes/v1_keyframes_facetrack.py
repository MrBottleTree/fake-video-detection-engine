import cv2
import os
import numpy as np

def run(state: dict) -> dict:
    print("Node V1: Extracting keyframes and tracking faces...")
    output_dir = state.get("data_dir")
    debug = state.get("debug", False)
    
    if not output_dir or not os.path.exists(output_dir):
        print(f"Error: Data directory not found at {output_dir}")
        return state

    try:
        keyframes_dir = os.path.join(output_dir, "keyframes")
        faces_dir = os.path.join(output_dir, "faces")
        os.makedirs(keyframes_dir, exist_ok=True)
        os.makedirs(faces_dir, exist_ok=True)

        cap = cv2.VideoCapture(os.path.join(output_dir, "video.mp4"))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {os.path.join(output_dir, "video.mp4")}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_area = frame_width * frame_height
        
        if debug:
            print(f"[DEBUG] V1: Video FPS: {fps}, Size: {frame_width}x{frame_height}")
        
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        prototxt_path = os.path.join(model_dir, "deploy.prototxt")
        model_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        
        def download_model_file(url, path):
            if not os.path.exists(path):
                print(f"Downloading {os.path.basename(path)}...")
                import urllib.request
                urllib.request.urlretrieve(url, path)
                print(f"Downloaded {path}")

        download_model_file(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            prototxt_path
        )
        download_model_file(
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            model_path
        )

        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        
        # Attempt to use CUDA
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            
            # Run a dummy forward pass to check if CUDA works
            dummy_blob = np.zeros((1, 3, 300, 300), dtype=np.float32)
            net.setInput(dummy_blob)
            net.forward()
            
            print("Node V1: Face Detection running on CUDA")
            if debug:
                print("[DEBUG] V1: CUDA backend successfully initialized.")
        except Exception as e:
            print(f"Node V1: Face Detection running on CPU (CUDA failed or unavailable: {e})")
            if debug:
                print(f"[DEBUG] V1: CUDA backend failed ({e}), falling back to CPU.")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        keyframes_paths = []
        face_detections = []
        
        current_time = 0.0
        frame_count = 0
        
        while True:
            frame_id = int(current_time * fps)
            if frame_id >= total_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            
            if not ret:
                break
            

            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            
            keyframe_filename = f"frame_{frame_id:06d}.jpg"
            keyframe_path = os.path.join(keyframes_dir, keyframe_filename)
            cv2.imwrite(keyframe_path, frame)
            keyframes_paths.append(keyframe_path)
            
            detections_in_frame = []
            face_list = []

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
                
                if w <= 0 or h <= 0:
                    continue
                    
                face_list.append({
                    "x": startX, "y": startY, "w": w, "h": h, 
                    "area": w * h, "confidence": float(confidence)
                })
            
            face_list.sort(key=lambda f: f["area"], reverse=True)
            
            for i, face in enumerate(face_list):
                x, y, w, h = face["x"], face["y"], face["w"], face["h"]
                area = face["area"]
                
                if area < (frame_area * 0.005):
                    continue
                
                is_main = (i == 0)
                
                pad_w = int(w * 0.2)
                pad_h = int(h * 0.2)
                
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(frame_width, x + w + pad_w)
                y2 = min(frame_height, y + h + pad_h)
                
                face_crop = frame[y1:y2, x1:x2]
                
                face_filename = f"face_{frame_id:06d}_{i}.jpg"
                face_path = os.path.join(faces_dir, face_filename)
                cv2.imwrite(face_path, face_crop)
                
                detections_in_frame.append({
                    "bbox": {"x": x, "y": y, "w": w, "h": h},
                    "confidence": face["confidence"],
                    "is_main": is_main,
                    "crop_path": face_path
                })
            
            face_detections.append({
                "frame_id": frame_id,
                "timestamp": current_time,
                "faces": detections_in_frame,
                "keyframe_path": keyframe_path
            })
            
            frame_count += 1
            current_time += 1.0

        cap.release()
        
        print(f"Extracted {len(keyframes_paths)} keyframes.")
        
        state["keyframes"] = keyframes_paths
        state["face_detections"] = face_detections
        
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"]["video_fps"] = fps
        state["metadata"]["total_frames"] = total_frames
        state["metadata"]["face_detection_model"] = "opencv_dnn_ssd"
        
        if debug:
            print(f"[DEBUG] V1: Saved keyframes to {keyframes_dir}")
            print(f"[DEBUG] V1: Saved face crops to {faces_dir}")
            total_faces = sum(len(d['faces']) for d in face_detections)
            print(f"[DEBUG] V1: Detected {total_faces} valid faces")

    except Exception as e:
        print(f"Error in V1 node: {e}")
        raise e

    return state
