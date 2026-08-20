import torch
import torch.nn as nn


class EEGCNN(nn.Module):

    def __init__(
        self,
        num_channels=9,
        num_classes=2
    ):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                num_channels,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.4),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


if __name__ == "__main__":

    from src.utils.device import get_device

    device = get_device()

    model = EEGCNN(
        num_channels=9,
        num_classes=2
    ).to(device)

    # Our verified STFT shape:
    # channels × frequency × time
    x = torch.randn(
        4,
        9,
        129,
        59,
        device=device
    )

    y = model(x)

    print(
        "Input:",
        x.shape
    )

    print(
        "Output:",
        y.shape
    )

    print(
        "Model device:",
        next(
            model.parameters()
        ).device
    )

    print(
        "\nCNN GPU test successful."
    )