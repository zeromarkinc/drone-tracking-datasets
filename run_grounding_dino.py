import os

import cv2
import numpy as np
import torch
from groundingdino.util.inference import (
    annotate,
    batch_predict,
    load_image,
    load_model,
    predict,
)
from tqdm import tqdm


def get_labels_and_images(label_path: str, image_path: str):
    """
    Iterates through the labels dir and if it has a .txt file it checks image dir
    for the same name and appends the full path to each
    """
    labels = []
    images = []
    for label in os.listdir(label_path):
        base_name = os.path.splitext(label)[0]
        image_file = os.path.join(image_path, f"{base_name}.jpg")
        if os.path.exists(image_file):
            labels.append(label)
            images.append(image_file)
    return labels, images


def read_labels(label_path: str) -> list[list[float]]:
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            labels.append([float(x) for x in line.strip().split(" ")[1:]])
    return labels


class DroneDataset:
    def __init__(
        self,
        label_path: str,
        image_path: str,
        im_size: tuple[int, int] = (750, 1180),
        first_half: bool = True,
    ):
        self.label_path = label_path
        self.image_path = image_path
        self.labels, self.images = get_labels_and_images(label_path, image_path)
        self.im_size = im_size
        self.first_half = first_half
        
        if first_half:
            self.labels = self.labels[: len(self.labels) // 2]
            self.images = self.images[: len(self.images) // 2]
        else:
            self.labels = self.labels[len(self.labels) // 2:]
            self.images = self.images[len(self.images) // 2:]
    
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_path, self.labels[index])
        image_path = self.images[index]
        label = read_labels(label_path)
        _, image = load_image(image_path)
        image = torch.Tensor(image)

        image = torch.nn.functional.interpolate(image.unsqueeze(0), self.im_size).squeeze(0)
        image_path = image_path.split("/")[-1]
        return image, label, image_path

    @staticmethod
    def collate_fn(batch):
        images, labels, image_paths = zip(*batch)

        return torch.stack(images), labels, image_paths


def process_output(
    boxes: torch.Tensor, logits: torch.Tensor, old_box: list[float]
) -> list[float]:
    """
    Process the output of the grounding dino model
    """
    # if there are multiple take the top logit
    boxes = torch.stack(boxes)
    logits = torch.stack(logits)
    if len(boxes) > 1:
        error = 0
        best = np.inf
        for box in boxes:
            error += abs(box[0] - old_box[0])
            error += abs(box[1] - old_box[1])
            error += abs(box[2] - old_box[2])
            error += abs(box[3] - old_box[3])
            if error < best:
                best = error
                best_box = box
        box = best_box
    else:
        box = boxes[0]

    error = 0
    for i in range(len(box)):
        error += abs(box[i] - old_box[i])
    if box[2] < .03 or box[3] < .03:
        # we do not want to label exceptionally small drones right now
        return None
    if error > 0.1:
        return old_box
    return box.tolist()


def write_to_label_file(label_path: str, box: torch.Tensor):
    """
    Write the output to a label file
    """
    with open(label_path, "w") as f:
        f.write(f"0 {box[0]} {box[1]} {box[2]} {box[3]}\n")


def run_grounding_dino(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    output_label_path: str,
    text_prompt: str = "Drone flying",
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: str = "cuda:0",
):
    for images, labels, image_paths in tqdm(dataloader, desc="Running Grounding Dino Predictions"):
        images = images.to(device)
        boxes, logits, boxes_to_im = batch_predict(
            model=model,
            preprocessed_images=images,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )
        for im_num in range(len(images)):
            output_path = os.path.join(
                output_label_path, image_paths[im_num].replace(".jpg", ".txt")
            )
            final_boxes = []
            final_logits = []
            for i in range(len(boxes_to_im)):
                if boxes_to_im[i] == im_num:
                    final_boxes.append(boxes[i])
                    final_logits.append(logits[i])
            if len(final_boxes) == 0:
                # i think if grounding dino doesnt detect it we should not be labeling it
                # as these drones would be exceptionally far away
                continue  
            else: 
                box = process_output(final_boxes, final_logits, old_box=labels[im_num][0])
                if box is not None:
                    write_to_label_file(output_path, box)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-first_half", action="store_true")
    parser.add_argument("-second_half", action="store_true")
    args = parser.parse_args()
    if args.first_half:
        DEVICE = "cuda"
        dataset = DroneDataset("yolo_dataset/train/labels", "yolo_dataset/train/images", first_half=True)
    else:
        DEVICE = "cuda"
        dataset = DroneDataset("yolo_dataset/train/labels", "yolo_dataset/train/images", first_half=False)
    labels, images = get_labels_and_images(
        "yolo_dataset/train/labels", "yolo_dataset/train/images"
    )
    model_path = "/home/derek_austin/GroundingDINO/groundingdino_swint_ogc.pth"
    model_config_path = "/home/derek_austin/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    model = load_model(model_config_path, model_path).to(DEVICE)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=8,
        collate_fn=dataset.collate_fn,
    )
    run_grounding_dino(
        model, dataloader, "yolo_dataset/grounding_dino_labels", device=DEVICE
    )
