import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import albumentations as A
import cv2
class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = sorted(os.listdir(os.path.join(data_dir, "images")))
        self.masks = sorted(os.listdir(os.path.join(data_dir, "masks")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.data_dir, "images", img_name)
        mask_name = self.masks[idx]
        mask_path = os.path.join(self.data_dir, "masks", mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        
        if self.transform:
            transformed_data = self.transform(image=np.array(image), mask=np.array(mask))
            image = transformed_data['image']
            mask = transformed_data['mask']

        return np.array(image).astype(np.float32), np.array(mask).astype(np.float32)
