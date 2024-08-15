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
import torchvision
from torch.utils.tensorboard import SummaryWriter
from utils import (
    one_hot_encode_images,
    log_segmentation_results,
    calculate_metrics,
    map_one_hot_to_image,
)
from PIL import Image


# Hyperparameters
num_classes = 11  # 20 classes + background
batch_size = 8
learning_rate = 0.01
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Device configuration


train_dataset = SegmentationDataset(data_dir="./fcn_data")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
resize = transforms.Resize((640, 640))
model = FCN(3, num_classes).to(device)  # Initialize the model
try:

    model.load_state_dict(torch.load("fcn_model.pth"))
    print("loaded model")
except Exception as e:
    pass
# Initialize model, loss, and optimizer
criterion = SegmentationLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

color_map = torch.tensor(
    [
        [0, 0, 0],
        [25, 82, 255],
        [255, 25, 197],
        [140, 255, 25],
        [0, 0, 0],
        [226, 255, 25],
        [255, 197, 25],
        [140, 25, 255],
        [54, 255, 25],
        [25, 255, 82],
        [255, 111, 25],
    ]
)

print("training started")
counter = 0

# Setup variables
log_dir = "./logs"  # Directory to save TensorBoard logs
writer = SummaryWriter(log_dir=log_dir)


# Training loop
for epoch in range(num_epochs):
    try:
        
        model.train()
        running_loss = 0.0
        iou_total = 0.0
        dice_total = 0.0
        precision_total = 0.0
        recall_total = 0.0
    
        for i, (images, masks) in enumerate(train_loader):
            optimizer.zero_grad()
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
    
            masks = one_hot_encode_images(masks,color_map)
    
            outputs = model(images)
    
            # predicted_mask = torch.argmax(outputs, dim=1)  # Shape: [1, 640, 640]
    
            # # Convert to RGB image using the color map
            # mask_rgb = color_map[predicted_mask].float().requires_grad_()
            loss = criterion(outputs,masks)
            # loss = criterion(mask_rgb.float(), masks.float())
            running_loss += loss.item()
    
            loss.backward()
            optimizer.step()
    
            # Calculate metrics
            iou, dice, precision, recall = calculate_metrics(outputs, masks)
            iou_total += iou
            dice_total += dice
            precision_total += precision
            recall_total += recall
            print(f"{i+1}/{len(train_loader)}")
    
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
    
        image=map_one_hot_to_image(outputs.permute(0, 3, 2, 1), color_map)[0]            
        image = image.numpy()
        image = (image * 255).astype(np.uint8)
    
        # Plot the image
        plt.imshow(image)
        plt.show()
        scheduler.step()
        torch.save(model.state_dict(), "fcn_model.pth")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"error in epoch {epoch}, {e}")
# Close the writer after training
writer.close()
# Save the model
torch.save(model.state_dict(), "fcn_model.pth")
