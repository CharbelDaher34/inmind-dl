import torch
import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # FCN layers
        self.fcn = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, kernel_size=1),
        )

        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.ConvTranspose2d(
                num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.ConvTranspose2d(
                num_classes,
                num_classes,
                kernel_size=16,
                stride=8,
                padding=4,
                bias=False,
            ),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.conv4(x)
        x = nn.ReLU()(x)
        x = self.conv5(x)
        x = self.fcn(x)
        x = self.upsample(x)
        return x
