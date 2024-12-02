import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm


def read_detections(label_path):
    detections = {}
    with open(label_path, 'r') as file:
        for line in file:
            try:    
                frame, x, y = map(float, line.split())
                detections[int(frame)] = (x, y)
            except ValueError:
                print(f"Skipping line: {line}")
    return detections

def convert_to_yolo_format(x, y, w, h):
    return f"0 {x} {y} {w} {h}"

def process_dataset(dataset, cam_id, output_folder, default_box_size=0.05):
    detection_file = f"datasets/detections/{dataset}/{cam_id}.txt"
    frames_folder = f"datasets/{dataset}_{cam_id}"
    
    if not os.path.exists(detection_file):
        print(f"Detection file not found: {detection_file}")
        return
    
    if not os.path.exists(frames_folder):
        print(f"Frames folder not found: {frames_folder}")
        return
    
    output_image_folder = os.path.join(output_folder, "images")
    output_label_folder = os.path.join(output_folder, "labels")
    
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)
    
    detections = read_detections(detection_file)
    # open the first frame to get the image size
    frame_path = os.path.join(frames_folder, f"frame_{1:04d}.jpg")
    frame = cv2.imread(frame_path)
    height, width = frame.shape[:2]
    
    for frame_number in tqdm(range(1, len(os.listdir(frames_folder)) + 1), desc=f"Processing {dataset}/{cam_id}"):
        frame_path = os.path.join(frames_folder, f"frame_{frame_number:04d}.jpg")
        
        if not os.path.exists(frame_path):
            print(f"Frame not found: {frame_path}")
            continue
        
        # check to see detections is not 0, 0
        if frame_number in detections and detections[frame_number] == (0, 0):
            continue
        
        # Copy image to output folder
        output_image_path = os.path.join(output_image_folder, f"{dataset}_{cam_id}_frame_{frame_number:04d}.jpg")
        shutil.copy2(frame_path, output_image_path)
        
        # Create YOLO format annotation
        if frame_number in detections:
            x, y = detections[frame_number]
            x = x / width
            y = y / height
            yolo_annotation = convert_to_yolo_format(x, y, default_box_size, default_box_size)
            
            output_label_path = os.path.join(output_label_folder, f"{dataset}_{cam_id}_frame_{frame_number:04d}.txt")
            with open(output_label_path, 'w') as f:
                f.write(yolo_annotation)

if __name__ == "__main__":
    datasets = ["dataset1", "dataset2", "dataset3", "dataset4"]
    cam_ids = ["cam0", "cam1", "cam2", "cam3", "cam4", "cam5", "cam6"]
    datasets = ["dataset3"]
    cam_ids = ["cam0"]
    
    output_folder = "yolo_dataset"
    
    os.makedirs(output_folder, exist_ok=True)
    
    for dataset in datasets:
        for cam_id in cam_ids:
            process_dataset(dataset, cam_id, output_folder)

    print("Conversion to YOLO format completed.")
    import os
    os.system("mkdir yolo_dataset/train")
    os.system("mv yolo_dataset/images yolo_dataset/train")
    os.system("mv yolo_dataset/labels yolo_dataset/train")
