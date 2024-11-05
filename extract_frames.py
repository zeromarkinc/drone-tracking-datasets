import os
import shutil

import cv2
import numpy as np
import tqdm


def copy_to_datasets_folder(source_folder, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)
    for item in os.listdir(source_folder):
        s = os.path.join(source_folder, item)
        d = os.path.join(destination_folder, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

if __name__ == "__main__":
    datasets = ["dataset1", "dataset2", "dataset3", "dataset4"]
    cam_ids = ["cam0", "cam1", "cam2", "cam3", "cam4", "cam5", "cam6"]
    
    # Copy frames to datasets/frames
    os.makedirs("datasets/frames", exist_ok=True)
    for dataset in datasets:
        for cam_id in cam_ids:
            source_folder = f"{dataset}/{cam_id}_frames"
            if os.path.exists(source_folder):
                destination_folder = f"datasets/frames/{dataset}_{cam_id}"
                copy_to_datasets_folder(source_folder, destination_folder)
    
    # Copy detections to datasets/detections
    os.makedirs("datasets/detections", exist_ok=True)
    for dataset in datasets:
        source_folder = f"{dataset}/detections"
        destination_folder = f"datasets/detections/{dataset}"
        if os.path.exists(source_folder):
            copy_to_datasets_folder(source_folder, destination_folder)
