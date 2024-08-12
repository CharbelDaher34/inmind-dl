import os
import torch
from PIL import Image
import json
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_names = sorted(os.listdir(data_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        number = image_name[4:8]

        # Load image
        image_path = os.path.join(self.data_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        # Load bounding box label
        bounding_box_label_path = os.path.join(
            self.data_dir, f"bounding_box_2d_tight_labels_{number}.json"
        )
        with open(bounding_box_label_path, "r") as f:
            bouding_box_labels = json.load(f)

        # Load tight prim paths (assuming you need it)
        tight_prim_path_path = os.path.join(
            self.data_dir, f"bounding_box_2d_tight_prim_paths_{number}.json"
        )
        with open(tight_prim_path_path, "r") as f:
            tight_prim_paths = json.load(f)

        # Load segmentation image
        segmentation_image_path = os.path.join(
            self.data_dir, f"semantic_segmentation_{number}.png"
        )
        segmentation_image = Image.open(segmentation_image_path)

        # Load segmentation labels
        segmentation_label_path = os.path.join(
            self.data_dir, f"semantic_segmentation_labels_{number}.json"
        )
        with open(segmentation_label_path, "r") as f:
            segmentation_labels = json.load(f)

        # Preprocess data (example)
        if self.transform:
            image = self.transform(image)
        # Convert bounding box to tensor or desired format
        bouding_box_labels = torch.tensor(bouding_box_labels, dtype=torch.float32)
        # Preprocess other data as needed

        return (
            image,
            bouding_box_labels,
            tight_prim_paths,
            segmentation_image,
            segmentation_labels,
        )
