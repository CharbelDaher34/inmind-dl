import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import FCN
from loss import SegmentationLoss
from dataset import SegmentationDataset
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

# Hyperparameters
num_classes = 11  # 20 classes + background
batch_size = 8
learning_rate = 0.1
num_epochs = 20

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = SegmentationDataset(data_dir="./fcn_data")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
resize = transforms.Resize((640, 640))
model = FCN(3, num_classes).to(device)  # Initialize the model
try:

    model.load_state_dict(torch.load("fcn_model.pth"))
except Exception as e:
    pass
# Initialize model, loss, and optimizer
criterion = SegmentationLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
# Color map: Define a color for each class (R, G, B)
# color_map = torch.tensor(
#     [
#         [0, 0, 0],  # Class 0: Black
#         [128, 0, 0],  # Class 1: Maroon
#         [0, 128, 0],  # Class 2: Green
#         [128, 128, 0],  # Class 3: Olive
#         [0, 0, 128],  # Class 4: Navy
#         [128, 0, 128],  # Class 5: Purple
#         [0, 128, 128],  # Class 6: Teal
#         [128, 128, 128],  # Class 7: Gray
#         [255, 0, 0],  # Class 8: Red
#         [0, 255, 0],  # Class 9: Lime
#         [0, 0, 255],  # Class 10: Blue
#     ]
# )
color_map = torch.tensor(
    [
        [0, 0, 0, 0],
        [25, 82, 255, 255],
        [255, 25, 197, 255],
        [140, 255, 25, 255],
        [0, 0, 0, 255],
        [226, 255, 25, 255],
        [255, 197, 25, 255],
        [140, 25, 255, 255],
        [54, 255, 25, 255],
        [25, 255, 82, 255],
        [255, 111, 25, 255],
    ]
)
color_map = color_map[:, :3]

print("training started")
counter = 0
from PIL import Image

# Setup variables
log_dir = "./logs"  # Directory to save TensorBoard logs
writer = SummaryWriter(log_dir=log_dir)


# Example function to calculate metrics
def calculate_metrics(predicted_masks, true_masks):
    smooth = 1e-6

    intersection = (predicted_masks & true_masks).float().sum((1, 2))
    union = (predicted_masks | true_masks).float().sum((1, 2))
    iou = (intersection + smooth) / (union + smooth)

    dice = (
        2
        * intersection
        / (
            predicted_masks.float().sum((1, 2))
            + true_masks.float().sum((1, 2))
            + smooth
        )
    )

    tp = (predicted_masks & true_masks).float().sum((1, 2))
    fp = (predicted_masks & ~true_masks).float().sum((1, 2))
    fn = (~predicted_masks & true_masks).float().sum((1, 2))

    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)

    return (
        iou.mean().item(),
        dice.mean().item(),
        precision.mean().item(),
        recall.mean().item(),
    )


# Training loop
for epoch in range(num_epochs):
    model.train()
    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)
        images = images.permute(0, 3, 2, 1)
        images = resize(images)
        images = images.permute(0, 3, 2, 1)
        images = images[:, :, :, :3]  # Remove the 4th channel if it's not needed

        masks = masks.permute(0, 3, 2, 1)
        masks = resize(masks)
        masks = masks.permute(0, 3, 2, 1)
        masks = masks[:, :, :, :3]  # Remove the 4th channel if it's not needed

        outputs = model(images)

        predicted_mask = torch.argmax(outputs, dim=1)  # Shape: [1, 640, 640]
        # Convert to RGB image using the color map
        mask_rgb = color_map[predicted_mask].float().requires_grad_()
        loss = criterion(mask_rgb.float(), masks.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("1batch")
    plt.plot(predicted_mask[0])
    plt.show()
    scheduler.step()
    torch.save(model.state_dict(), "fcn_model.pth")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), "fcn_model.pth")
