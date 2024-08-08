import albumentations as A
import cv2
from model import UNET
from dataset import unetDataset
from utils import (
    get_loaders,
    save_predictions_as_imgs,
    check_accuracy,
    save_checkpoint,
    load_checkpoint,
)
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


##Data loading
trainTransform = A.Compose(
    [
        A.Resize(256, 256),  # Resize images and masks to 256x256
        A.RandomCrop(224, 224),  # Randomly crop to 224x224
        A.HorizontalFlip(p=0.5),  # 50% chance to flip horizontally
        A.VerticalFlip(p=0.5),  # 50% chance to flip vertically
        A.RandomRotate90(p=0.5),  # 50% chance to rotate 90 degrees
        A.Normalize(),  # Normalization
    ],
)
valTransform = A.Compose(
    [
        A.Normalize(),
    ],
)
train_loader, val_loader = get_loaders(
    "./train",
    "./train_masks",
    "val_images",
    "val_masks",
    4,
    trainTransform,
    valTransform,
    num_workers=4,
    pin_memory=True,
)
train_loader = tqdm(train_loader, desc="Training", unit="batch")
val_loader = tqdm(val_loader, desc="Validation", unit="batch")

### Setup variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss = nn.BCEWithLogitsLoss()
try:

    model = UNET()
    load_checkpoint("my_checkpoint.pth.tar", model)
except Exception as e:
    model = UNET()

optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 2


### Create the training loop
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    try:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)

                images = images.permute(0, 3, 1, 2)
                masks = masks.unsqueeze(1).float()  # Adding channel dimension

                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(train_loader)
            scheduler.step()

            # Validation phase
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    images = images.permute(0, 3, 1, 2)
                    masks = masks.unsqueeze(1).float()  # Adding channel dimension

                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item() * images.size(0)

            val_loss /= len(val_loader)

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            print()
            print()
            # save checkpoint, this save must be done if the new checkpoint is better than previous but for the sake of simplicity i will keep it like this
            save_checkpoint(model.state_dict(), filename="my_checkpoint.pth.tar")
    except Exception as e:
        print(f"Error encountered during training: {e}")


# Call the training function
train(model, train_loader, val_loader, loss, optimizer, num_epochs, device)

# Output
### Display a confirmation message
print("Training completed.")


save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=device)
check_accuracy(val_loader, model)
