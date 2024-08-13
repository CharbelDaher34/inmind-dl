import torch
import torch.nn as nn
import torchvision.models as models


class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()

        # Load a pretrained VGG16 model
        vgg16 = models.vgg16(pretrained=True)

        # Use the features of VGG16
        self.features = vgg16.features

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
        x = self.features(x)
        x = self.fcn(x)
        x = self.upsample(x)
        return x
