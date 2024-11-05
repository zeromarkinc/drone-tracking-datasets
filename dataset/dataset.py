"""
Creating a dataloader that assumes the data is in a folder with the following structure (yolo format):

test/
    images/
        image1.jpg
    labels/
        image1.txt
train/
    images/
        image1.jpg
    labels/
        image1.txt
val/
    images/
        image1.jpg
    labels/
        image1.txt
"""

import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2.functional._geometry import crop_bounding_boxes, resize

# from .transforms import rt_detr_video_train_transform




import random
import torch
from torchvision.transforms import v2
from torchvision.transforms.v2.functional._geometry import resize
from typing import Tuple


def rt_detr_train_transform(
    img_size: int = 640,
) -> v2.Compose:
    return v2.Compose(
        [
            v2.Resize((img_size, img_size)),
            v2.RandomZoomOut(side_range=(1, 2)),
            v2.RandomPhotometricDistort(
                brightness=(0.5, 1.5),
                contrast=(0.5, 1.5),
                saturation=(0.5, 1.5),
                hue=(-0.25, 0.25),
                p=0.7,
            ),
            v2.RandomIoUCrop(
                min_scale=0.2,
                max_scale=1.0,
                min_aspect_ratio=0.5,
                max_aspect_ratio=2.0,
                sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                trials=40,
            ),
            v2.SanitizeBoundingBoxes(min_area=12),
            v2.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 3.0)),
            v2.Resize((img_size, img_size)),
            v2.RandomGrayscale(p=0.2),
            v2.RandomHorizontalFlip(),
            v2.RandomInvert(p=0.2),
            v2.ToTensor(),
        ]
    )
    

def rt_detr_val_transform(
    img_size: int = 640,
) -> v2.Compose:
    return v2.Compose(
        [
            v2.Resize((img_size, img_size)),
            v2.ToTensor(),
        ]
    )

# def rt_detr_train_transform(
#     img_size: int = 640,
# ) -> v2.Compose:
#     return v2.Compose(
#         [
#             v2.Resize((img_size, img_size)),
#             v2.RandomZoomOut(side_range=(1, 2)),
#             v2.RandomPhotometricDistort(
#                 brightness=(0.875, 1.125),
#                 contrast=(0.5, 1.5),
#                 saturation=(0.5, 1.5),
#                 hue=(-0.05, 0.05),
#             ),
#             v2.RandomIoUCrop(
#                 min_scale=0.2,
#                 max_scale=1.0,
#                 min_aspect_ratio=0.5,
#                 max_aspect_ratio=2.0,
#                 sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
#                 trials=40,
#             ),
#             v2.SanitizeBoundingBoxes(min_area=12),
#             v2.GaussianBlur(kernel_size=(9, 9), sigma=(.1, 3.0)),
#             v2.Resize((img_size, img_size)),
#             v2.RandomHorizontalFlip(),
#             v2.ToTensor(),
#         ]
#     )
    

class customFrameJitter(torch.nn.Module):
    """To simulate the jittering of frames in a video"""
    def __init__(self, p: float = 0.5, pixel_jitter: int = 5):
        super().__init__()
        self.p = p
        self.pixel_jitter = pixel_jitter

    def forward(self, x: torch.Tensor):
        original_shape = x[0][0].shape
        for i in range(len(x[0])):
            if random.random() < self.p:
                shape = x[0][i].shape
                start_jitter = random.randint(0, self.pixel_jitter)
                end_jitter = None
                end_jitter = -random.randint(0, self.pixel_jitter)
                end_jitter = None if end_jitter == 0 else end_jitter
                x[0][i] = x[0][i][:, start_jitter:end_jitter, start_jitter:end_jitter]
                if i == 0:
                    end_jitter = 0 if end_jitter is None else end_jitter
                    x[1]['boxes'][:, [0, 1]] = x[1]['boxes'][:, [0, 1]] - torch.Tensor([start_jitter, start_jitter])
                    x[1]['boxes'].canvas_size = (shape[-2] - start_jitter + end_jitter, shape[-1] - start_jitter + end_jitter)
                    for i in range(x[1]['boxes'].shape[0]):
                        box = x[1]['boxes'][i]
                        box = self.clean_box(box, (shape[-2] - start_jitter + end_jitter, shape[-1] - start_jitter + end_jitter))
                        x[1]['boxes'][i] = torch.tensor(box)
                else:
                    x[0][i] = resize(x[0][i], (original_shape[-2], original_shape[-1]))
        return x
    
    def clean_box(self, box: torch.Tensor, image_shape: Tuple[int, int]):
        """
        This take a cxcywh box and if w or h go out of bounds adjust center
        and clips them to be in bounds
        """
        cx, cy, w, h = box
        if cx - w/2 < 0:
            right_most_spot = cx + w/2
            if right_most_spot < image_shape[1]:
                w = right_most_spot
                cx = w / 2
            else:
                w = image_shape[1]
                cx = image_shape[1] / 2
        elif cx + w/2 > image_shape[1]:
            left_most_spot = cx - w/2
            if left_most_spot < image_shape[1]:
                w = image_shape[1] - left_most_spot
                cx = left_most_spot + w/2
            else:
                w = 0
                cx = 0
            
        if cy - h/2 < 0:
            bottom_most_spot = cy + h/2
            if bottom_most_spot > 0:
                h = bottom_most_spot
                cy = h / 2
            else:
                h = image_shape[0]
                cy = image_shape[0] / 2
        elif cy + h/2 > image_shape[0]:
            top_most_spot = cy - h/2
            if top_most_spot < image_shape[0]:
                h = image_shape[0] - top_most_spot
                cy = top_most_spot + h / 2
            else:
                h = 0
                cy = 0
        return cx, cy, w, h
    


def rt_detr_video_train_transform(
    img_size: int = 256,
    heavy_augs: bool = False,
) -> v2.Compose:
    if heavy_augs:
        return v2.Compose(
                [
                    v2.Resize((img_size, img_size)),
                    v2.RandomZoomOut(side_range=(1, 2)),
                    v2.RandomPhotometricDistort(
                        brightness=(0.5, 1.5),
                        contrast=(0.5, 1.5),
                        saturation=(0.5, 1.5),
                        hue=(-0.25, 0.25),
                        p=0.7,
                    ),
                    v2.RandomIoUCrop(
                        min_scale=0.2,
                        max_scale=1.0,
                        min_aspect_ratio=0.5,
                        max_aspect_ratio=2.0,
                        sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                        trials=40,
                    ),
                    v2.SanitizeBoundingBoxes(min_area=12),
                    v2.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 3.0)),
                    v2.RandomGrayscale(p=0.2),
                    v2.RandomHorizontalFlip(),
                    v2.RandomInvert(p=0.2),
                    v2.Resize((img_size, img_size)),
                    v2.ToTensor(),
                    customFrameJitter(p=0.9999, pixel_jitter=10),
                    v2.SanitizeBoundingBoxes(min_area=12),
                    v2.Resize((img_size, img_size)),
                ]
            )
    else:
        return v2.Compose(
            [
                v2.Resize((img_size, img_size)),
                v2.RandomZoomOut(side_range=(1, 2), p=.1),
                v2.RandomPhotometricDistort(
                    brightness=(0.9, 1.1),
                    contrast=(0.9, 1.1),
                    saturation=(0.9, 1.1),
                    hue=(-0.05, 0.05),
                    p=0.4,
                ),
                v2.RandomIoUCrop(
                    min_scale=0.3,
                    max_scale=1.0,
                    min_aspect_ratio=0.5,
                    max_aspect_ratio=2.0,
                    sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                    trials=40,
                ),
                v2.SanitizeBoundingBoxes(min_area=12),
                v2.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 1.0)),
                v2.RandomGrayscale(p=0.1),
                v2.RandomHorizontalFlip(),
                v2.RandomInvert(p=0.05),
                v2.Resize((img_size, img_size)),
                v2.ToTensor(),
                customFrameJitter(p=1.0, pixel_jitter=5),
                v2.SanitizeBoundingBoxes(min_area=12),
                v2.Resize((img_size, img_size)),
            ]
        )
    
    
class resizeVideoTransform(torch.nn.Module):
    def __init__(self, img_size: int = 256):
        """The pytorch transform resizes only the first frame so this resizes the other frames"""
        super().__init__()
        self.img_size = img_size
        
    def forward(self, x: torch.Tensor):
        for i in range(len(x[0])):
            if i != 0:
                x[0][i] = resize(x[0][i], (self.img_size, self.img_size))
        return x
    
def rt_detr_video_val_transform(
    img_size: int = 256,
) -> v2.Compose:
    return v2.Compose(
        [v2.Resize((img_size, img_size)), resizeVideoTransform(img_size), v2.ToTensor()]
    )



class VideoDTDYoloFormatDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split: str = "train",
        transform: v2.Compose = rt_detr_video_train_transform,
        num_samples: int = None,
        empty_labels: bool = True,
    ):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform

        self.path_to_images = os.path.join(self.root_dir, "images")
        self.path_to_labels = os.path.join(self.root_dir, "labels")
        self.images_paths = os.listdir(self.path_to_images)[:10]
        self.labels_paths = os.listdir(self.path_to_labels)

        # create the image path to label path dict
        self.image_to_label = {}
        self.video_paths = set()
        print("Reading images")
        for img in tqdm.tqdm(self.images_paths):
            file_ext = os.path.splitext(img)[1]
            img_base = os.path.basename(img)[: len(file_ext) * -1]
            if img_base in self.image_to_label:
                raise ValueError(f"Duplicate image {img_base} found")
            if f"{img_base}.txt" in self.labels_paths:
                self.image_to_label[img] = f"{img_base}.txt"
            elif f"{img_base}_synthetic.txt" in self.labels_paths:
                self.image_to_label[img] = f"{img_base}_synthetic.txt"
            else:
                raise ValueError(f"Label for image {img_base} not found")
            assert (
                f"{img_base}.txt" in self.labels_paths
                or f"{img_base}_synthetic.txt" in self.labels_paths
            ), (
                f"Label for image {img_base} not found"
                f"Label path: {os.path.join(self.root_dir, 'labels', self.image_to_label[img])}"
            )
            self.sort_into_tuples(img)
        print(f"Number of videos: {len(self.video_paths)}")

        # assert len(self.images_paths) == len(
        #     self.labels_paths
        # ), "Number of images and labels must be the same"
        if not empty_labels:
            self.remove_empty_images()
        self.images_paths = self.images_paths[:num_samples]
        self.video_paths = list(self.video_paths)

    def sort_into_tuples(self, image: str):
        """Sort the images and labels into tuples for video loading"""

        label_path = image.replace(self.path_to_images, self.path_to_labels)
        last_segment_length = len(image.split("_")[-1])
        frame_number = int(image.split("_")[-1].split(".")[0])
        image_base = image[: -last_segment_length - 1]
        label_path = label_path.replace(".jpg", ".txt")
        current_tuple = (
            f"{image_base}_{frame_number - 2:04d}.jpg",
            f"{image_base}_{frame_number - 1:04d}.jpg",
            f"{image_base}_{frame_number:04d}.jpg",
            f"{label_path}",  # we only care about the last frame labels
        )
        # check to make sure all exist otherwise skip
        if (
            not os.path.exists(os.path.join(self.path_to_images, current_tuple[0]))
            or not os.path.exists(os.path.join(self.path_to_images, current_tuple[1]))
            or not os.path.exists(os.path.join(self.path_to_images, current_tuple[2]))
        ):
            # print(f"Skipping {current_tuple} because it does not exist")
            return
        self.image_to_label[current_tuple[0]] = current_tuple[3]
        self.image_to_label[current_tuple[1]] = current_tuple[3]
        self.image_to_label[current_tuple[2]] = current_tuple[3]
        if current_tuple in self.video_paths:
            return
        self.video_paths.add(current_tuple)

    def remove_empty_images(self):
        """Function to remove those images without any labels"""
        total_removed = 0
        imgs_to_remove = set()
        for image1, image2, image3, label in self.video_paths:
            label_path = os.path.join(
                self.root_dir, "labels", self.image_to_label[image1]
            )
            with open(label_path, "r") as f:
                labels = f.readlines()
            if len(labels) == 0:
                imgs_to_remove.add((image1, image2, image3, label))
        for img in imgs_to_remove:
            self.video_paths.remove(img)
            total_removed += 1
        print(f"Removed {total_removed} images without labels")
        print(f"Total images: {len(self.images_paths)}")

    def __len__(self):
        return len(self.video_paths)

    def get_labels(self, label_path: str, height: int, width: int):
        with open(label_path, "r") as f:
            labels = f.readlines()

        if len(labels) == 0:
            # Handle empty labels file
            labels = []
            bboxes = tv_tensors.BoundingBoxes(
                torch.empty((0, 4)),
                format=tv_tensors.BoundingBoxFormat.CXCYWH,
                canvas_size=(height, width),
            )
            final_annotation = {
                "boxes": bboxes,
                "labels": torch.empty(0, dtype=torch.long),
            }
        else:
            labels = [label.strip().split() for label in labels]
            labels = [
                [
                    int(label[0]),
                    float(label[1]) * width,
                    float(label[2]) * height,
                    float(label[3]) * width,
                    float(label[4]) * height,
                ]
                for label in labels
            ]
            bboxes = tv_tensors.BoundingBoxes(
                torch.Tensor(
                    [[label[1], label[2], label[3], label[4]] for label in labels]
                ),
                format=tv_tensors.BoundingBoxFormat.CXCYWH,
                canvas_size=(height, width),
            )
            final_annotation = {
                "boxes": bboxes,
                "labels": torch.Tensor([label[0] for label in labels]).long(),
            }
        return final_annotation

    def __getitem__(self, idx):
        image_path1, image_path2, image_path3, label_path = self.video_paths[idx]
        img1 = Image.open(os.path.join(self.root_dir, "images", image_path1)).convert(
            "RGB"
        )
        img2 = Image.open(os.path.join(self.root_dir, "images", image_path2)).convert(
            "RGB"
        )
        img3 = Image.open(os.path.join(self.root_dir, "images", image_path3)).convert(
            "RGB"
        )
        height = img1.height
        width = img1.width

        final_annotation = self.get_labels(
            os.path.join(self.root_dir, "labels", label_path), height, width
        )

        imgs = torch.stack(
            [
                torch.Tensor(np.array(img1)),
                torch.Tensor(np.array(img2)),
                torch.Tensor(np.array(img3)),
            ]
        )
        imgs, labels = self.transform(([img3, img2, img1], final_annotation))
        height = imgs[0][1].shape[-2]
        width = imgs[0][1].shape[-1]
        if len(labels["boxes"]) > 0:
            labels["boxes"][:, [0, 2]] /= width
            labels["boxes"][:, [1, 3]] /= height
        labels["img_name"] = self.images_paths[idx]
        imgs = torch.stack([imgs[0], imgs[1], imgs[2]])
        sample = {
            "image": imgs.view(9, 256, 256).contiguous(),
            "annotations": labels,
        }
        return sample

    @staticmethod
    def collate_fn(batch: List[dict]):
        images = [item["image"] for item in batch]
        annotations = [item["annotations"] for item in batch]
        return torch.stack(images), annotations







if __name__ == "__main__":
    # example
    dataset = VideoDTDYoloFormatDataset(
        "/Users/derek/Desktop/drone-tracking-datasets/drone_tracking_dataset_yolo_format",
        "train",
        transform=rt_detr_video_train_transform(),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=dataset.collate_fn,
    )
    for batch in dataloader:
        import pdb; pdb.set_trace()
        break
