import os
import subprocess
import re
import json

VIDEO_DIR = "videos"
RESULTS_FILE = "batch_results.txt"

def get_videos():
    return [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]

def get_label(filename):
    filename_lower = filename.lower()
    if "deepfake" in filename_lower or "fake" in filename_lower or "ai" in filename_lower or "synthesia" in filename_lower:
        return 1
    return 0

import argparse

def run_pipeline(video_file, train_mode=False):
    video_path = os.path.join(VIDEO_DIR, video_file)
    label = get_label(video_file)
    print(f"Processing {video_file} (Label: {label})...")
    
    try:
        # Run main.py and capture output
        cmd = ["uv", "run", "python", "main.py", "--debug", video_path]
        if train_mode:
            cmd.append(str(label))
            
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
        return None, label

def main():
    parser = argparse.ArgumentParser(description="Batch Test Video Pipeline")
    parser.add_argument("--train", action="store_true", help="Pass labels to main.py to enable online training")
    args = parser.parse_args()

    videos = get_videos()
    results = {}
    
    print(f"Found {len(videos)} videos.")
    print(f"Training Mode: {'ENABLED' if args.train else 'DISABLED'}")
    
    with open(RESULTS_FILE, "w") as f:
        f.write(f"{'Video Name':<60} | {'Label':<5} | {'Prob':<10} | {'Pred':<5} | {'Status'}\n")
        f.write("-" * 100 + "\n")
    
    correct_count = 0
    total_processed = 0

    for video in videos:
        prob, label = run_pipeline(video, train_mode=args.train)
        
        if prob is not None:
            total_processed += 1
            prediction = 1 if prob > 0.5 else 0
            pred_str = "FAKE" if prediction == 1 else "REAL"
            label_str = "FAKE" if label == 1 else "REAL"
            
            is_correct = (prediction == label)
            if is_correct:
                correct_count += 1
            
            status = "CORRECT" if is_correct else "WRONG"
            
            color_code = "\033[92m" if is_correct else "\033[91m"
            reset_code = "\033[0m"
            
            print(f"{color_code}{video:<40} | L:{label} | P:{prob:.20f} | {status}{reset_code}")
            
            with open(RESULTS_FILE, "a") as f:
                f.write(f"{video:<60} | {label:<5} | {prob:.20f}     | {pred_str:<5} | {status}\n")
        else:
             with open(RESULTS_FILE, "a") as f:
                f.write(f"{video:<60} | {label:<5} | {'ERROR':<10} | {'N/A':<5} | ERROR\n")

    if total_processed > 0:
        accuracy = (correct_count / total_processed) * 100
        print(f"\nTotal Accuracy: {accuracy:.2f}% ({correct_count}/{total_processed})")
        with open(RESULTS_FILE, "a") as f:
            f.write(f"\nTotal Accuracy: {accuracy:.2f}% ({correct_count}/{total_processed})\n")

if __name__ == "__main__":
    main()
