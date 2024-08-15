import torch.nn as nn


class ClothingClassifierModel(nn.Module):
    def __init__(self):
        super(ClothingClassifierModel, self).__init__()
        self.image_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.LeakyReLU(),
            nn.Linear(1568 * 2, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.image_layer(x)
        return x
