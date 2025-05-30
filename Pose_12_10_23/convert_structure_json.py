import json
import pickle
import numpy as np
import os


def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(i) for i in obj]
    else:
        return obj
    

def json_to_pkl(json_path, pkl_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"File JSON convertito in PKL: {pkl_path}")


def pkl_to_json(pkl_path, json_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    # Converte tutto in strutture serializzabili
    data = convert_ndarray_to_list(data)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"File PKL convertito in JSON: {json_path}")



file = "47"

# Percorsi file
input_path = f"results_{file}.json"

# Ricarichiamo il file originale
with open(input_path, "r") as f:
    data = json.load(f)

# Ricostruzione output con struttura corretta
output_corrected = {
    "keypoints": [],
    "scores": [],
    "w_h": [850, 720]
}

# Iterazione e costruzione nuova struttura
for frame in data["instance_info"]:
    frame_kps = []
    frame_scores = []
    for instance in frame["instances"]:
        kps = instance["keypoints"]
        scores = instance["keypoint_scores"]
        frame_kps.append([[x, y] for x, y in kps])
        frame_scores.append(scores)  # lista piatta
    output_corrected["keypoints"].append(frame_kps)
    output_corrected["scores"].append(frame_scores)

# Salvataggio file corretto
corrected_output_path = f"{file}.json"
with open(corrected_output_path, "w") as f:
    json.dump(output_corrected, f, indent=4)




input = f"{file}.json"
output = f"{file}.pkl"



json_to_pkl(input, output)
#pkl_to_json(input, output)