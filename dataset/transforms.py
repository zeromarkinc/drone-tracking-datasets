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
