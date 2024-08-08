import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def convert_mask(mask_path):
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)
    mask_np[mask_np == 255] = 1
    return Image.fromarray(mask_np)


class unetDataset(Dataset):
    def __init__(self, mask_dir, image_dir, transform=None):
        self.mask_dir = mask_dir
        self.mask_path = sorted(os.listdir(mask_dir))

        self.image_dir = image_dir
        self.img_path = sorted(os.listdir(image_dir))

        self.transform = transform

    def __len__(self):
        return len(self.mask_path)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.img_path[idx])
        image = Image.open(img_path).convert("RGB")

        mask_path = os.path.join(self.mask_dir, self.mask_path[idx])
        mask = convert_mask(mask_path)

        try:
            transformed = self.transform(image=np.array(image), mask=np.array(mask))
            image = transformed["image"]
            mask = transformed["mask"]
        except KeyError as e:
            print(f"Key error: {e}")
        return torch.from_numpy(image), torch.from_numpy(mask)
