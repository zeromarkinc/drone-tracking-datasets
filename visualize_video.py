import cv2
import numpy as np
from tqdm import tqdm


def read_detections(label_path: str) -> list:
    detections_dict = {}
    with open(label_path, 'r') as file:
        for line in file:
            try:
                frame, x, y = map(float, line.split())
                detections_dict[frame] = (x, y)
            except:
                print(line)
                continue
            
    return detections_dict



if __name__ == "__main__":
    cam_number = 6
    dataset_number = 4
    FRAME_PATH = f"/Users/derek/Desktop/drone-tracking-datasets/dataset{dataset_number}/cam{cam_number}_frames"
    VIDEO_PATH = f"/Users/derek/Desktop/drone-tracking-datasets/dataset{dataset_number}/cam{cam_number}.mp4"
    DETECTION_PATH = f"/Users/derek/Desktop/drone-tracking-datasets/dataset{dataset_number}/detections/cam{cam_number}.txt"

    data = read_detections(DETECTION_PATH)

    video_capture = cv2.VideoCapture(VIDEO_PATH)
    FPS = video_capture.get(cv2.CAP_PROP_FPS)

    FRAME_WIDTH = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video = cv2.VideoWriter(
        f"cam{cam_number}_tracking.mp4",
        cv2.VideoWriter_fourcc(*"XVID"),
        FPS,
        (FRAME_WIDTH, FRAME_HEIGHT),
    )
    keys = sorted(list(data.keys()))[-3000:]
    offset = 0
    
    for i in tqdm(range(len(keys))):
        try:
            image = cv2.imread(f"{FRAME_PATH}/frame_{int(keys[i]):04d}.jpg")
            key = keys[i] + offset
            x, y = data[key]
            if x != 0 or y != 0:
                cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)
            output_video.write(image)
        except:
            print(f"Error at frame {key}")
            continue
    output_video.release()
