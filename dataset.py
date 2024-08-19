import os
import torch
from PIL import Image
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_names = sorted(
            [
                name
                for name in os.listdir(data_dir)
                if name.startswith("rgb_") and name.endswith(".png")
            ]
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        number = image_name[4:8]

        image_path = os.path.join(self.data_dir, image_name)
        # image = Image.open(image_path).convert("RGB")

        bounding_box_label_path = os.path.join(
            self.data_dir, f"bounding_box_2d_tight_{number}.npy"
        )
        bounding_boxes = np.load(bounding_box_label_path)
        # Check and convert bounding boxes to a supported type if necessary
        yoloBoxes = []
        for box_data in bounding_boxes:
            class_id = box_data["semanticId"]
            x_center = (box_data["x_min"] + box_data["x_max"]) / 2
            y_center = (box_data["y_min"] + box_data["y_max"]) / 2
            width = box_data["x_max"] - box_data["x_min"]
            height = box_data["y_max"] - box_data["y_min"]
            # Normalize to [0, 1] based on image dimensions (assuming image size is known)
            image_width, image_height = (
                1280,
                720,
            )  # Replace with actual image dimensions
            x_center /= image_width
            y_center /= image_height
            width /= image_width
            height /= image_height
            yoloBoxes.append(
                [
                    int(class_id),
                    float(x_center),
                    float(y_center),
                    float(width),
                    float(height),
                ]
            )
        segmentation_image_path = os.path.join(
            self.data_dir, f"semantic_segmentation_{number}.png"
        )
        # segmentation_image = Image.open(segmentation_image_path)

        # if self.transform:
        #     image = self.transform(image)

        return (
            os.path.abspath(image_path),
            yoloBoxes,
            os.path.abspath(segmentation_image_path),
        )
