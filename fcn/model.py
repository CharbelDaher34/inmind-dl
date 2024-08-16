import torch
from torch import nn
class FCN(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(FCN, self).__init__()
        self.n_classes=n_classes
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third convolutional block
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fourth convolutional block
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu10 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fifth convolutional block
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu12 = nn.ReLU(inplace=True)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu13 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers (implemented as convolutions)
        self.conv14 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.relu14 = nn.ReLU(inplace=True)
        self.conv15 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu15 = nn.ReLU(inplace=True)

        # Final classification layer
        self.conv16 = nn.Conv2d(4096, n_classes, kernel_size=1)

        # Upsampling layers
        self.up1 = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=4, stride=2, padding=1
        )
        self.up2 = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=4, stride=2, padding=1
        )
        self.up3 = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=16, stride=8, padding=4
        )
        self.skip_conv1 = nn.Conv2d(256, self.n_classes, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(512, self.n_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)  # Add softmax layer

    def forward(self, x):
        # x=x.permute(0,3,1,2)
        # Encoder
        x = self.relu2(self.conv2(self.relu1(self.conv1(x))))
        x = self.pool1(x)

        x = self.relu4(self.conv4(self.relu3(self.conv3(x))))

        x = self.pool2(x)
        x = self.relu7(self.conv7(self.relu6(self.conv6(self.relu5(self.conv5(x))))))
        x = self.pool3(x)
        skip1 = self.skip_conv1(x)

        x = self.relu10(self.conv10(self.relu9(self.conv9(self.relu8(self.conv8(x))))))
        x = self.pool4(x)

        skip2 = self.skip_conv2(x)

        x = self.relu13(
            self.conv13(self.relu12(self.conv12(self.relu11(self.conv11(x)))))
        )
        x = self.pool5(x)

        # Fully connected layers
        x = self.relu15(self.conv15(self.relu14(self.conv14(x))))

        # Final classification layer
        x = self.conv16(x)

        # Upsampling and skip connections
        x = self.up1(x)

        # print(f"x:{x.shape}")
        # print(f"skip2:{skip2.shape}")
        # return
        x = x + skip2
        x = self.up2(x)
        x = x + skip1
        x = self.up3(x)
        return x


# model=FCN(3,11)
# # x=torch.randn(1,640,640,3)
# # print(model(x))
# import torch
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # Sample image tensor
# images = torch.randn(1, 1280, 720, 3)
# masks = torch.randn(1, 1280, 720, 3)
# resize = transforms.Resize((640, 640))

# masks = masks.to(device)


# images = images.to(device)
# masks = masks.to(device)
# image = images.permute(0, 3, 2, 1)
# images = resize(image)
# images = images.permute(0, 3, 2, 1)

# masks=masks.permute(0,3,2,1)
# masks = resize(masks)
# masks = masks.permute(0,3,2,1)
# print(f"masks:{masks.shape}")
# print(f"images:{images.shape}")

# output = model(images)
# print(output)

# # Color map: Define a color for each class (R, G, B)
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

# ## Model
# # Get the predicted class for each pixel
# predicted_mask = torch.argmax(output, dim=1)  # Shape: [1, 640, 640]

# # Convert to RGB image using the color map
# mask_rgb = color_map[predicted_mask]  # Shape: [1, 640, 640, 3]

# ## Output
# # The final 3-channel mask
# print(mask_rgb.shape)  # Should be [1, 640, 640, 3]
# ## Output
# # Squeeze to remove the batch dimension and plot the image
# mask_rgb = mask_rgb.squeeze(0)  # Shape: [640, 640, 3]

# plt.imshow(mask_rgb.numpy().astype(int))
# plt.axis("off")  # Hide axes
# plt.show()
