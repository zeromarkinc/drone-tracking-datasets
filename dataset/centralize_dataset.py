import os
import shutil
from typing import Tuple
import tqdm

import cv2


def read_detections(label_path: str) -> list:
    detections_dict = {}
    with open(label_path, "r") as file:
        for line in file:
            try:
                frame, x, y = map(float, line.split())
                detections_dict[frame] = (x, y)
            except:
                print(line)
                continue

    return detections_dict


def write_detections(
    detection: Tuple, label_path: str, h: float = 0.05, w: float = 0.05
):
    x, y = detection
    with open(label_path, "a") as file:
        file.write(f"0 {x} {y} {h} {w}\n")


if __name__ == "__main__":
    # crawl all dataset{x}/cam{num}_frames and copy into central images folder
    # read the dataset{x}/detections/cam{num}.txt and rewrite to separate yolo compatible text files
    # using a standard h and w for the drone of .05

    central_image_path = "/Users/derek/Desktop/drone-tracking-datasets/drone_tracking_dataset_yolo_format/train/images"
    central_label_path = "/Users/derek/Desktop/drone-tracking-datasets/drone_tracking_dataset_yolo_format/train/labels"
    datasets = [1, 2, 3, 4]
    cam_ids = [0, 1, 2, 3, 4, 5, 6]
    broken_dataset_cam_id = [(3, 1)]

    for dataset in datasets:
        for cam_id in tqdm.tqdm(cam_ids):
            if (dataset, cam_id) in broken_dataset_cam_id:
                continue

            path_to = f"dataset{dataset}/cam{cam_id}_frames"
            if len(os.listdir(path_to)) == 0:
                continue

            detection_path = f"dataset{dataset}/detections/cam{cam_id}.txt"
            detections = read_detections(detection_path)
            
            height_of_cam = None
            width_of_cam = None
            for file in os.listdir(path_to):
                if height_of_cam is None or width_of_cam is None:
                    height_of_image, width_of_image, _ = cv2.imread(os.path.join(path_to, file)).shape
                    if height_of_cam is None:
                        height_of_cam = height_of_image
                    if width_of_cam is None:
                        width_of_cam = width_of_image
                
                try:
                    # rename to include the dataset and cam number in the image name
                    new_file_name = f"dataset{dataset}_cam{cam_id}_{file}"
                    frame_number = int(file.split("_")[1].split(".")[0])
                    detection = detections[frame_number]
                    
                    detection_x = detection[0] / width_of_cam
                    detection_y = detection[1] / height_of_cam
                    if detection_x == 0 or detection_y == 0:
                        continue
                    shutil.copy2(
                        os.path.join(path_to, file),
                        os.path.join(central_image_path, new_file_name),
                    )
                    write_detections(
                        (detection_x, detection_y),
                        os.path.join(central_label_path, new_file_name.replace(".jpg", ".txt")),
                    )
                except Exception as e:
                    print(f"Error at frame {frame_number}, error: {e}")
                    continue
