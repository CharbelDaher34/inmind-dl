"""
Main file for training Yolo model on Pascal VOC dataset

"""

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc. 
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "./data/images"
LABEL_DIR = "./data/labels"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])





def main():

    train_dataset = VOCDataset(
        "./data/100examples.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = VOCDataset(
        "./data/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )


    # Setup model
    model=Yolov1(split_size=7, num_boxes=2, num_classes=20)
   

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(),lr=0.1)
    # Setup loss 
    loss = YoloLoss()
    # Create train fn
    def train_one_epoch(model, optimizer, loss, dataloader):
        model.train()
        epoch_loss = 0.0
        for batch_data, batch_targets in dataloader:
            optimizer.zero_grad()  # Zero the gradient buffers
            output = model(batch_data)  # Forward pass
            loss = loss(output, batch_targets)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            epoch_loss += loss.item()
        return epoch_loss / len(dataloader)
    # Create training loop
    for epoch in range(2):
        epoch_loss = train_one_epoch(model, optimizer, loss, train_loader)
        print(f"Epoch {epoch + 1}/{2}, Loss: {epoch_loss}")


if __name__ == "__main__":
    main()