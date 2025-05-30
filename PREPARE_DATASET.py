import os
import subprocess
import json
import shutil
import pickle
import numpy as np
from tqdm import tqdm

# === CONFIG ===
VIDEO_DIR = "input_videos"
LABEL_FILE = "labels.txt"
OUTPUT_DIR = "my_dataset_output"
OPENPOSE_BIN = "./build/examples/openpose/openpose.bin"  # aggiorna se serve

# === DESTINATION STRUCTURE ===
POSE_OUT = os.path.join(OUTPUT_DIR, "pose_format")
RGB_OUT = os.path.join(OUTPUT_DIR, "rgb_format")
LABEL_OUT = os.path.join(OUTPUT_DIR, "data", "train")
OPENPOSE_TMP = "openpose_tmp"

os.makedirs(POSE_OUT, exist_ok=True)
os.makedirs(RGB_OUT, exist_ok=True)
os.makedirs(LABEL_OUT, exist_ok=True)
os.makedirs(OPENPOSE_TMP, exist_ok=True)

def run_openpose(video_path, out_dir):
    subprocess.run([
        OPENPOSE_BIN,
        "--video", video_path,
        "--write_json", out_dir,
        "--hand", "--face",
        "--display", "0",
        "--render_pose", "0"
    ], check=True)

def extract_keypoints(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if len(data["people"]) == 0:
        return np.zeros((1, 133, 2)), np.zeros((1, 133))

    person = data["people"][0]
    body = np.array(person['pose_keypoints_2d']).reshape(-1, 3)  # 25
    face = np.array(person['face_keypoints_2d']).reshape(-1, 3)  # 70
    hand_l = np.array(person['hand_left_keypoints_2d']).reshape(-1, 3)  # 21
    hand_r = np.array(person['hand_right_keypoints_2d']).reshape(-1, 3)  # 21
    all_kp = np.concatenate([body, hand_l, hand_r, face], axis=0)  # [133, 3]

    return all_kp[:, :2][None], all_kp[:, 2][None]

def json_to_pkl(json_folder, out_path):
    keypoints = []
    scores = []
    jsons = sorted([f for f in os.listdir(json_folder) if f.endswith('.json')])
    for f in jsons:
        kp, sc = extract_keypoints(os.path.join(json_folder, f))
        keypoints.append(kp)
        scores.append(sc)

    with open(out_path, 'wb') as f:
        pickle.dump({
            "keypoints": keypoints,
            "scores": scores
        }, f)

def load_labels_grouped_by_video():
    grouped = []
    with open(LABEL_FILE, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if "--" in line]

    current_video = []
    last_index = 0
    for line in lines:
        parts = line.split("--")
        if len(parts) == 3:
            current_video.append(parts[2].strip())
        if len(current_video) > 0 and len(grouped) == last_index:
            grouped.append(current_video)
            current_video = []
            last_index += 1

    if current_video:
        grouped.append(current_video)

    return grouped

def main():
    all_labels = load_labels_grouped_by_video()
    label_json = []

    for idx, text_lines in tqdm(enumerate(all_labels), total=len(all_labels)):
        video_file = f"{idx}.mp4"
        video_path = os.path.join(VIDEO_DIR, video_file)
        openpose_json_dir = os.path.join(OPENPOSE_TMP, str(idx))
        os.makedirs(openpose_json_dir, exist_ok=True)

        print(f"[{idx}] Running OpenPose on {video_file}...")
        run_openpose(video_path, openpose_json_dir)

        print(f"[{idx}] Converting JSON to PKL...")
        pkl_path = os.path.join(POSE_OUT, f"{idx}.pkl")
        json_to_pkl(openpose_json_dir, pkl_path)

        print(f"[{idx}] Copying video...")
        shutil.copy(video_path, os.path.join(RGB_OUT, video_file))

        label_json.append({
            "video": video_file,
            "pose": f"{idx}.pkl",
            "text": " ".join(text_lines)
        })

    out_json_path = os.path.join(LABEL_OUT, "CSL_News_Labels.json")
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(label_json, f, indent=2, ensure_ascii=False)

    print("✅ Dataset creato con successo!")

if __name__ == "__main__":
    main()
