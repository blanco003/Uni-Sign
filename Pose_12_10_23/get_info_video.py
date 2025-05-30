import cv2
import sys

def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Errore nell'aprire il video: {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()
    print(f"Formato video: {height} x {width}")

input_video = "47.mp4"
get_video_resolution(input_video)
