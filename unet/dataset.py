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


class CustomImageDataset(Dataset):
    def __init__(
        self, mask_dir, img_dir, transform=None
    ):
        self.mask_dir = mask_dir
        self.mask_path = sorted(os.listdir(mask_dir))

        self.img_dir = img_dir
        self.img_path = sorted(os.listdir(img_dir))

        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_path[idx])
        image = Image.open(img_path).convert("RGB")

        mask_path = os.path.join(self.mask_dir, self.mask_path[idx])
        mask = convert_mask(mask_path)
        
        try:
            transformed = transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        except KeyError as e:
            print(f"Key error: {e}")
        return torch.from_numpy(image), torch.from_numpy(mask)


import albumentations as A
import cv2

transform = A.Compose(
    [
        A.Resize(256, 256),  # Resize images and masks to 256x256
        A.RandomCrop(224, 224),  # Randomly crop to 224x224
        A.HorizontalFlip(p=0.5),  # 50% chance to flip horizontally
        A.VerticalFlip(p=0.5),  # 50% chance to flip vertically
        A.RandomRotate90(p=0.5),  # 50% chance to rotate 90 degrees
        A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),  # Normalization
    ],
)
dataset = CustomImageDataset(
    mask_dir="./train_masks",img_dir="./train",transform=transform
)
