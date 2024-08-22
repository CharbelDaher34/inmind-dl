import pretty_errors
import warnings
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from fcnBackbone import FCN
from loss import SegmentationLoss
from dataset import SegmentationDataset
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utils import calculate_metrics
import tqdm
import traceback
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
import cv2

warnings.filterwarnings("ignore")

# Hyperparameters
num_classes = 10  # 9 classes + background
batch_size = 1
learning_rate = 1e-3
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
train_dataset = SegmentationDataset(
    data_dir="./fcn_data",
    transform=A.Compose(
        [
            A.Rotate(limit=45, interpolation=cv2.INTER_AREA),
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ]
    ),
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = FCN(3, num_classes).to(device)  # FCN expects 3-channel input images
try:
    model = torch.load("./bestModelFinalBack.pt").to(device)
    print("Loaded model")
except Exception as e:
    pass

# Initialize loss and optimizer
criterion = SegmentationLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=3, verbose=True
)


# Color map for converting RGB masks to class indices
color_map = {
    0: [0, 0, 0],
    1: [25, 82, 255],
    2: [255, 25, 197],
    3: [140, 255, 25],
    4: [226, 255, 25],
    5: [255, 197, 25],
    6: [140, 25, 255],
    7: [54, 255, 25],
    8: [25, 255, 82],
    9: [255, 111, 25],
}


def convert_rgb_to_class_index(mask, color_map=color_map):
    h, w, c = mask.shape
    mask_out = np.zeros((h, w), dtype=np.uint8)
    for class_idx, color in color_map.items():
        color = np.array(color)
        class_mask = np.all(mask == color, axis=-1)
        mask_out[class_mask] = class_idx
    return mask_out


def preprocess_batch_images(images):
    # Keep only the first 3 channels
    images = images[:, :, :, :3]
    images = images.to(device)
    images = images.float()  # Convert to float for division
    images /= 255.0
    # Permute to (batch_size, channels, height, width)
    images = images.permute(0, 3, 1, 2)
    # Resize images to (640, 640)
    images = F.interpolate(
        images, size=(640, 640), mode="bilinear", align_corners=False
    )
    images = images.permute(0, 2, 3, 1)

    return images


def preprocess_batch_masks(masks):
    # Keep only the first 3 channels
    masks = masks.float()
    masks = masks[:, :, :, :3]

    # Permute to (batch_size, channels, height, width)
    masks = masks.permute(0, 3, 1, 2)
    # Resize masks to (640, 640)
    masks = F.interpolate(masks, size=(640, 640), mode="nearest")

    # Convert RGB masks to class indices
    masks_out = [
        torch.from_numpy(
            convert_rgb_to_class_index(mask.permute(1, 2, 0).cpu().numpy())
        ).long()
        for mask in masks
    ]
    masks_out = torch.stack(masks_out).to(device)

    # Convert class indices to one-hot encoding
    masks_one_hot = (
        F.one_hot(masks_out, num_classes=num_classes).permute(0, 3, 1, 2).float()
    )
    masks = masks / 255.0

    return masks_one_hot, masks.permute(0, 2, 3, 1)


def convert_index_to_image(index, color_map):
    """
    Converts a batch of class probability tensors to images.

    Args:
        index: A torch tensor of shape [batch_size, num_classes, height, width] with class probabilities.
        color_map: A dictionary mapping class indices to color values.

    Returns:
        A torch tensor of shape [batch_size, height, width, 3] representing the images.
    """

    # Take the argmax over the class dimension (axis=1)
    max_class = torch.argmax(index, dim=1)  # Shape [batch_size, height, width]

    # Initialize an output image tensor with the same batch size, height, and width, but with 3 channels for RGB
    batch_size, height, width = max_class.shape
    output_image = torch.zeros((batch_size, height, width, 3), dtype=torch.uint8)

    # Vectorized mapping of class indices to RGB colors for each image in the batch
    for class_idx, color in color_map.items():
        output_image[max_class == class_idx] = torch.tensor(color, dtype=torch.uint8)

    return output_image


# Setup TensorBoard writer
log_dir = "./logss"
writer = SummaryWriter(log_dir=log_dir)
iou_total = 0
dice_total = 0
precision_total = 0
recall_total = 0
# Training loop
print("Training started")
for epoch in range(num_epochs):
    try:
        model.train()
        running_loss = 0.0

        for i, (images, masks) in tqdm.tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):

            images = preprocess_batch_images(images)
            index, masks = preprocess_batch_masks(masks)
            optimizer.zero_grad()  # Added this line

            outputs = model(images)
            loss = criterion(outputs, index)
            # loss = criterion(mask_rgb.float(), masks.float())
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            # Calculate metrics
            iou, dice, precision, recall = calculate_metrics(
                convert_index_to_image(outputs, color_map=color_map), masks
            )
            iou_total += iou
            dice_total += dice
            precision_total += precision
            recall_total += recall
            # print(f"{i+1}/{len(train_loader)}")
            if i % 100 == 0:
                torch.save(model, "bestModelBack.pt")

        # Log metrics to TensorBoard after each epoch
        avg_loss = running_loss / len(train_loader)
        avg_iou = iou_total / len(train_loader)
        avg_dice = dice_total / len(train_loader)
        avg_precision = precision_total / len(train_loader)
        avg_recall = recall_total / len(train_loader)

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Metrics/IoU", avg_iou, epoch)
        writer.add_scalar("Metrics/Dice", avg_dice, epoch)
        writer.add_scalar("Metrics/Precision", avg_precision, epoch)
        writer.add_scalar("Metrics/Recall", avg_recall, epoch)

        # Visualization once per epoch
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        axes[0].imshow(masks[0].cpu().numpy())
        axes[0].set_title("Ground Truth")
        outputsRgb = convert_index_to_image(outputs, color_map=color_map)
        axes[1].imshow(outputsRgb[0].detach().cpu().numpy())
        axes[1].set_title("Prediction")
        plt.savefig(f"segmentation_result_epoch.png")
        plt.close()

        # Save the model once per epoch
        torch.save(model, f"bestModelBack.pt")

        # Step the scheduler
        scheduler.step(avg_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    except Exception as e:
        print(f"--------------------Error in epoch {epoch}")
        traceback.print_exc()
        print("--------------------")

# Close the writer after training
writer.close()
# Save the model
torch.save(model.state_dict(), "fcn_modelBack.pth")
torch.save(model, "bestModelFinalBack.pt")
