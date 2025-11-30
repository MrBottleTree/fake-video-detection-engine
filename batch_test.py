import os
import subprocess
import re
import json

VIDEO_DIR = "videos"
RESULTS_FILE = "batch_results.txt"

import json
import math
import cv2

def get_video_duration(filename):
    path = os.path.join(VIDEO_DIR, filename)
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return float('inf')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else float('inf')
        cap.release()
        return duration
    except Exception:
        return float('inf')

def get_videos():
    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
    # Sort by duration
    videos.sort(key=get_video_duration)
    return videos

def get_label(filename):
    filename_lower = filename.lower()
    if "deepfake" in filename_lower or "fake" in filename_lower or "ai" in filename_lower or "synthesia" in filename_lower:
        return 1
    return 0

import argparse

def run_local_lr(video_file, train_mode=False, label=None):
    video_path = os.path.join(VIDEO_DIR, video_file)
    video_name = os.path.basename(video_path)
    feature_file = os.path.join("features", f"{video_name}.json")
    
    if not os.path.exists(feature_file):
        return None, None

    try:
        with open(feature_file, "r") as f:
            features = json.load(f)
        
        # Load weights
        weights = {}
        if os.path.exists("lr_weights.json"):
            with open("lr_weights.json", "r") as f:
                weights = json.load(f)
        
        # Calculate probability locally
        z = weights.get("bias", 0.0)
        for k, v in features.items():
            z += weights.get(k, 0.0) * float(v)
        
        if z >= 0:
            fake_prob = 1.0 / (1.0 + math.exp(-z))
        else:
            fake_prob = math.exp(z) / (1.0 + math.exp(z))
            
        # If training enabled and label provided, trigger online learning (update weights)
        if train_mode and label is not None:
             # Check if prediction is wrong before updating? 
             # The user logic implies we update if it's wrong, but the LR node updates on every sample usually.
             # However, for the "retry until correct" loop, we specifically want to update.
             # Let's update every time here to match LR node behavior, but the loop in main controls the retry.
             
             lr = 0.005
             error = fake_prob - float(label)
             weights["bias"] = weights.get("bias", 0.0) - lr * error * 1.0
             for k, v in features.items():
                 weights[k] = weights.get(k, 0.0) - lr * error * float(v)
             
             with open("lr_weights.json", "w") as f:
                 json.dump(weights, f, indent=2)
                 
        return fake_prob, label

    except Exception as e:
        print(f"Error in local LR: {e}")
        return None, None

def run_pipeline(video_file, train_mode=False, clear_cache=False):
    label = get_label(video_file)
    
    # Try local LR first if not clearing cache
    if not clear_cache:
        prob, _ = run_local_lr(video_file, train_mode, label)
        if prob is not None:
            return prob, label

    video_path = os.path.join(VIDEO_DIR, video_file)
    print(f"Processing {video_file} (Label: {label})...")
    
    try:
        # Run main.py and capture output
        cmd = ["uv", "run", "python", "main.py", "--debug", video_path]
        if train_mode:
            cmd.append(str(label))
        if clear_cache:
            cmd.append("--clear-cache")
            
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        output = result.stdout
        match = re.search(r"Fake probability: ([0-9e\.-]+)", output)
        if match:
            prob = float(match.group(1))
            return prob, label
        else:
            print(f"Could not find probability in output for {video_file}")
            return None, label
            
    except subprocess.CalledProcessError as e:
        print(f"Error processing {video_file}: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None, label

def main():
    parser = argparse.ArgumentParser(description="Batch Test Video Pipeline")
    parser.add_argument("--train", action="store_true", help="Pass labels to main.py to enable online training")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cached features before processing")
    args = parser.parse_args()

    videos = get_videos()
    results = {}
    
    print(f"Found {len(videos)} videos.")
    print(f"Training Mode: {'ENABLED' if args.train else 'DISABLED'}")
    print(f"Cache Clearing: {'ENABLED' if args.clear_cache else 'DISABLED'}")
    
    with open(RESULTS_FILE, "w") as f:
        f.write(f"{'Video Name':<60} | {'Label':<5} | {'Prob':<10} | {'Pred':<5} | {'Status'}\n")
        f.write("-" * 100 + "\n")
    
    correct_count = 0
    total_processed = 0

    for video in videos:
        max_retries = 100
        attempt = 0
        should_clear = args.clear_cache
        
        while True:
            prob, label = run_pipeline(video, train_mode=args.train, clear_cache=should_clear)
            should_clear = False # Only clear on first attempt
            
            if prob is not None:
                prediction = 1 if prob > 0.5 else 0
                is_correct = (prediction == label)
                
                if is_correct or not args.train or attempt >= max_retries:
                    # Log result and move to next video
                    total_processed += 1
                    if is_correct:
                        correct_count += 1
                    
                    pred_str = "FAKE" if prediction == 1 else "REAL"
                    status = "CORRECT" if is_correct else "WRONG"
                    color_code = "\033[92m" if is_correct else "\033[91m"
                    reset_code = "\033[0m"
                    
                    print(f"{color_code}{video:<40} | L:{label} | P:{prob:.20f} | {status}{reset_code}")
                    
                    with open(RESULTS_FILE, "a") as f:
                        f.write(f"{video:<60} | {label:<5} | {prob:.20f}     | {pred_str:<5} | {status}\n")
                    break
                else:
                    # Wrong prediction in training mode, retry
                    attempt += 1
                    print(f"\033[93m{video}: Prediction WRONG (P:{prob:.4f}, L:{label}). Retraining (Attempt {attempt}/{max_retries})...\033[0m")
            else:
                 with open(RESULTS_FILE, "a") as f:
                    f.write(f"{video:<60} | {label:<5} | {'ERROR':<10} | {'N/A':<5} | ERROR\n")
                 break

    if total_processed > 0:
        accuracy = (correct_count / total_processed) * 100
        print(f"\nTotal Accuracy: {accuracy:.2f}% ({correct_count}/{total_processed})")
        with open(RESULTS_FILE, "a") as f:
            f.write(f"\nTotal Accuracy: {accuracy:.2f}% ({correct_count}/{total_processed})\n")

if __name__ == "__main__":
    main()
