@echo off

python predict_single_video.py ^
    --video_path ./dataset/LIS/rgb_format/12_10_2023_0.mp4 ^
    --pose_path ./dataset/LIS/pose_format/12_10_2023_0.pkl
