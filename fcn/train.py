import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import FCN
from loss import SegmentationLoss
from dataset import SegmentationDataset
import numpy as np
# Hyperparameters
num_classes = 11  # 20 classes + background
batch_size = 1
learning_rate = 0.001
num_epochs = 2

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Data transforms
# transform = transforms.Compose(
#     [
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ]
# )

# Create datasets
train_dataset = SegmentationDataset(data_dir="./fcn_data")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss, and optimizer
model = FCN(num_classes).to(device)
criterion = SegmentationLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    print("training started")
    model.train()
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), "fcn_model.pth")
