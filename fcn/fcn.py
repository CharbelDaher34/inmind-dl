import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(FCN, self).__init__()
        self.n_classes = n_classes

        def conv_block(in_channels, out_channels, pool=False):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return nn.Sequential(*layers)

        # Encoder
        self.conv1 = conv_block(in_channels, 64, pool=True)
        self.conv2 = conv_block(64, 128, pool=True)
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.conv5 = conv_block(512, 512, pool=True)

        # Fully connected layers
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=1)
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

        # Skip connections
        self.skip_conv1 = nn.Conv2d(512, n_classes, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(256, n_classes, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        # Encoder
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        # Fully connected layers
        fc6 = F.relu(self.fc6(conv5))
        fc7 = F.relu(self.fc7(fc6))

        # Final classification layer
        score = self.score(fc7)

        # Upsampling and skip connections
        upscore2 = self.upsample1(score)
        skip1 = self.skip_conv1(conv4)
        upscore2 = upscore2 + skip1

        upscore4 = self.upsample2(upscore2)
        skip2 = self.skip_conv2(conv3)
        upscore4 = upscore4 + skip2

        upscore32 = self.upsample3(upscore4)

        return upscore32
