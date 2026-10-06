"""The two architectures. Both take a batch of grayscale faces of shape
(B, 1, 48, 48) with pixel values in [0, 1] and return 8 logits, so the
training loop and the webcam demo can treat them the same way."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from src.config import NUM_CLASSES

# Mean/std of FER2013 training pixels (scaled to [0, 1]); used by the scratch CNN.
FER_MEAN, FER_STD = 0.508, 0.254
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def conv_block(c_in, c_out, dropout):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout(dropout),
    )


class ScratchCNN(nn.Module):
    """A small VGG-style network trained from random weights.
    48x48 -> 24 -> 12 -> 6 -> 3, then global average pooling."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 64, 0.1),
            conv_block(64, 128, 0.1),
            conv_block(128, 256, 0.2),
            conv_block(256, 512, 0.2),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.5), nn.Linear(512, num_classes))

    def forward(self, x):
        x = (x - FER_MEAN) / FER_STD
        return self.head(self.features(x))


class ResNet18FER(nn.Module):
    """ResNet18 pretrained on ImageNet, with a new 8-class output layer.
    ImageNet models expect larger color images, so we upscale the 48x48
    grayscale face and copy it into 3 channels."""

    def __init__(self, num_classes=NUM_CLASSES, input_size=112, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = resnet18(weights=weights)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
        self.input_size = input_size
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        x = x.expand(-1, 3, -1, -1)
        return self.net((x - self.mean) / self.std)


def build_model(name: str, pretrained=True) -> nn.Module:
    if name == "cnn":
        return ScratchCNN()
    if name == "resnet18":
        return ResNet18FER(pretrained=pretrained)
    raise ValueError(f"Unknown model: {name}")
