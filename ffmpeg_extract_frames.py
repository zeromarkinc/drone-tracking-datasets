import os

import cv2

if __name__ == "__main__":
    command = 'ffmpeg -i {dataset_video_path} -qscale:v 2 -vf "fps={fps}" {output_folder}/frame_%04d.jpg'

    datasets = ["dataset1", "dataset2", "dataset3", "dataset4"]
    cam_ids = ["cam0", "cam1", "cam2", "cam3", "cam4", "cam5", "cam6"]

    for dataset in datasets:
        for cam_id in cam_ids:
            dataset_video_path = f"{dataset}/{cam_id}.mp4"
            output_folder = f"{dataset}/{cam_id}_frames"
            os.makedirs(output_folder, exist_ok=True)
            if not os.path.exists(dataset_video_path):
                print(f"Video file not found: {dataset_video_path}")
                continue
            video = cv2.VideoCapture(dataset_video_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            os.system(
                command.format(
                    dataset_video_path=dataset_video_path,
                    fps=fps,
                    output_folder=output_folder,
                )
            )
