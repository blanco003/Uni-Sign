import os
import torch
import argparse
from models import Uni_Sign
from datasets import S2T_Dataset_news, LIS_Dataset
from transformers import MT5Tokenizer

def predict_video(video_path: str, pose_path: str, model_ckpt: str, tokenizer_path: str, label_json_path: str, use_rgb: bool = True) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenizer e modello
    tokenizer = MT5Tokenizer.from_pretrained(tokenizer_path)
    model = Uni_Sign()
    checkpoint = torch.load(model_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    # Argomenti dummy per dataset
    class DummyArgs:
        dataset = "LIS"
        rgb_support = use_rgb
        data_transform = None

    args = DummyArgs()

    # Inizializza dataset per usare load_pose
    dataset = LIS_Dataset(
        path={args.dataset: label_json_path},
        args=args,
        phase="test"
    )

    # Carica i dati
    pose_sample, support_rgb_dict = dataset.load_pose(pose_path, os.path.basename(video_path))
    pose_sample = pose_sample.unsqueeze(0).to(device)

    # Predizione
    with torch.no_grad():
        output = model.generate(
            pose_sample=pose_sample,
            support_rgb_dict=support_rgb_dict,
            rgb_support=args.rgb_support,
            task="SLT",
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

    # Decodifica
    prediction = tokenizer.decode(output[0], skip_special_tokens=True)
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predizione singolo video con Uni-Sign")
    parser.add_argument("--video_path", type=str, required=True, help="Percorso al file .mp4 del video")
    parser.add_argument("--pose_path", type=str, required=True, help="Percorso al file .pkl dei keypoint")


    args = parser.parse_args()

    result = predict_video(
        video_path=args.video_path,
        pose_path=args.pose_path,
        model_ckpt=args.model_ckpt,
        tokenizer_path=args.tokenizer_path,
        label_json_path=args.label_json_path,
        use_rgb=not args.no_rgb
    )

    print(f"\n📝 Predizione: {result}")
