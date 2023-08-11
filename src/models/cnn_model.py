import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, num_classes=10 , dropout=0.5):  # adjust the number of classes
        super(CNNModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 196, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(196, 256, kernel_size=2, stride=1),
            # nn.ReLU(inplace=True),
            nn.BatchNorm2d(196),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        # calculate shape by doing one forward pass
        with torch.no_grad():
            dummy = torch.ones(1, 1, 128, 128)
            self.feature_dim = self._get_conv_out(dummy).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def _get_conv_out(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
