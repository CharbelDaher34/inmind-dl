import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


class FCN(nn.Module):
    def __init__(self, in_channels, n_classes, backbone="resnet50"):
        super(FCN, self).__init__()
        self.n_classes = n_classes

        # Load pre-trained ResNet backbone
        if backbone == "resnet50":
            resnet = models.resnet50(pretrained=True)
        elif backbone == "resnet101":
            resnet = models.resnet101(pretrained=True)
        else:
            raise ValueError("Unsupported backbone")

        # Encoder (use ResNet layers)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # Adjust the first convolutional layer if in_channels != 3
        if in_channels != 3:
            self.encoder[0] = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Fully connected layers
        self.fc6 = nn.Conv2d(2048, 4096, kernel_size=1)
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)

        # Final classification layer
        self.score = nn.Conv2d(4096, n_classes, kernel_size=1)

        # Upsampling layers
        self.upsample1 = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=4, stride=2, padding=1
        )
        self.upsample2 = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=4, stride=2, padding=1
        )
        self.upsample3 = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=16, stride=8, padding=4
        )
        self.trans_conv = nn.ConvTranspose2d(
            in_channels=10,  # Number of input channels
            out_channels=10,  # Number of output channels
            kernel_size=4,  # Kernel size for the transposed convolution
            stride=2,  # Stride to achieve upsampling by 2
            padding=1,  # Padding to maintain output size after upsampling
        )
        # Skip connections
        self.skip_conv1 = nn.Conv2d(2048, n_classes, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(1024, n_classes, kernel_size=1)

    def forward(self, x):
        # input_size = x.size()[2:]

        # Encoder
        features = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in {6, 7}:  # Save features after layer3 and layer4
                features.append(x)

        # Fully connected layers
        fc6 = F.relu(self.fc6(x))
        fc7 = F.relu(self.fc7(fc6))

        # Final classification layer
        score = self.score(fc7)

        # Upsampling and skip connections
        upscore2 = self.upsample1(score)
        skip1 = self.skip_conv1(features[1])
        upscore2 = upscore2[:, :, : skip1.size(2), : skip1.size(3)] + skip1

        upscore4 = self.upsample2(upscore2)

        skip2 = self.skip_conv2(features[0])
        upscore4 = upscore4[:, :, : skip2.size(2), : skip2.size(3)] + skip2
        upscore32 = self.upsample3(upscore4)
        # Ensure the output size matches the input size
        upscore64 = self.trans_conv(upscore32)
        return upscore64


# # Create an instance of the FCN model
# NUM_CLASSES = 10  # 21 classes for PASCAL VOC dataset
# model = FCN(in_channels=3, n_classes=NUM_CLASSES, backbone="resnet50")
# model.eval()  # Set the model to evaluation mode


# # Load and preprocess the image
# def preprocess_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     preprocess = transforms.Compose(
#         [
#             transforms.Resize((640, 640)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ]
#     )
#     input_tensor = preprocess(image)
#     input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
#     return input_batch, image


# # Load and preprocess an image
# image_path = "./fcn_data/images/rgb_0000.png"
# input_batch, original_image = preprocess_image(image_path)
# print(f"Input shape: {input_batch.shape}")

# # Run the model
# with torch.no_grad():
#     output = model(input_batch)

# print(f"Output shape: {output.shape}")

# # Post-process the output
# output = output.squeeze()  # Remove batch dimension
# output = output.argmax(dim=0)  # Get the class with highest probability for each pixel
# output = output.cpu().numpy()  # Convert to numpy array

# # Visualize the result
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(original_image)
# plt.title("Original Image")
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.imshow(output, cmap="jet")
# plt.title("Segmentation Result")
# plt.axis("off")

# plt.tight_layout()
# plt.show()
