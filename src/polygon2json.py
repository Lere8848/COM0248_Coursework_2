import os
import pickle
import json
from utils import DATASET_PATHS_WITH_TABLE, DATASET_PATHS_WITHOUT_TABLE, MISSING_POLYGON_LABELS

def extract_folder_key(path):
    return os.path.normpath(path).split(os.sep)[-1]

def generate_polygon_label_dict():
    label_dict = {}

    # with polygon file
    for dataset_path in DATASET_PATHS_WITH_TABLE:
        label_path = os.path.join(dataset_path, "labels", "tabletop_labels.dat")
        img_path = os.path.join(dataset_path, "image")
        folder_key = extract_folder_key(dataset_path)

        try:
            with open(label_path, 'rb') as f:
                polygon_labels = pickle.load(f)
        except FileNotFoundError:
            print(f"[Warning] No label file in {dataset_path}, skipping...")
            continue

        img_names = sorted(os.listdir(img_path))

        for i, img_name in enumerate(img_names):
            frame_name = os.path.splitext(img_name)[0]

            # MISSING POLYGON LABELS
            if folder_key in MISSING_POLYGON_LABELS and frame_name in MISSING_POLYGON_LABELS[folder_key]:
                label_dict[frame_name] = None
                continue

            if i < len(polygon_labels):
                polygons = polygon_labels[i]
                if polygons and len(polygons) > 0:
                    label_dict[frame_name] = polygons  # polygon: [[[x1...], [y1...]]]
                else:
                    label_dict[frame_name] = None
            else:
                label_dict[frame_name] = None

    # without polygon file
    # CW2.pdf p.9: 'mit_gym_z_squash', 'harvard_tea_2' negative samples
    for dataset_path in DATASET_PATHS_WITHOUT_TABLE:
        img_path = os.path.join(dataset_path, "image")
        img_names = sorted(os.listdir(img_path))

        for img_name in img_names:
            frame_name = os.path.splitext(img_name)[0]
            label_dict[frame_name] = None

    return label_dict


if __name__ == '__main__':
    polygon_label_dict = generate_polygon_label_dict()
    output_path = os.path.join(".", "data", "polygon_label_dict.json") # to data folder
    with open(output_path, "w") as f:
        json.dump(polygon_label_dict, f, indent=2)
    print(f"{output_path} saved. Total {len(polygon_label_dict)} frames.")