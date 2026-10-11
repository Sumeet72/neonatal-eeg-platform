import torch
import torch.nn as nn


class EEGNet(nn.Module):

    def __init__(
        self,
        channels=9,
        classes=2,
        dropout=0.5
    ):

        super().__init__()

        self.features = nn.Sequential(

            # ------------------------------------------------
            # Temporal / frequency feature extraction
            # ------------------------------------------------

            nn.Conv2d(
                channels,
                16,
                kernel_size=(5, 5),
                padding=(2, 2),
                bias=False
            ),

            nn.BatchNorm2d(
                16
            ),

            nn.ELU(),

            nn.Dropout(
                dropout
            ),


            # ------------------------------------------------
            # Depthwise-style spatial processing
            # ------------------------------------------------

            nn.Conv2d(
                16,
                32,
                kernel_size=(3, 3),
                padding=(1, 1),
                groups=16,
                bias=False
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ELU(),

            nn.AvgPool2d(
                kernel_size=(2, 2)
            ),

            nn.Dropout(
                dropout
            ),


            # ------------------------------------------------
            # Separable feature extraction
            # ------------------------------------------------

            nn.Conv2d(
                32,
                32,
                kernel_size=(3, 3),
                padding=(1, 1),
                groups=32,
                bias=False
            ),

            nn.Conv2d(
                32,
                32,
                kernel_size=(1, 1),
                bias=False
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ELU(),

            nn.AvgPool2d(
                kernel_size=(2, 2)
            ),

            nn.Dropout(
                dropout
            )
        )


        # Adaptive pooling means the classifier does not
        # depend on the exact STFT dimensions.

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )


        self.classifier = nn.Linear(
            32,
            classes
        )


    def forward(
        self,
        x
    ):

        x = self.features(
            x
        )

        x = self.pool(
            x
        )

        x = torch.flatten(
            x,
            start_dim=1
        )

        x = self.classifier(
            x
        )

        return x


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model = EEGNet(
        channels=9,
        classes=2
    ).to(device)


    x = torch.randn(
        4,
        9,
        129,
        59,
        device=device
    )


    output = model(
        x
    )


    print(
        "Device:",
        device
    )

    print(
        "Input shape:",
        x.shape
    )

    print(
        "Output shape:",
        output.shape
    )

    print(
        "EEGNet parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    assert output.shape == (
        4,
        2
    )

    print(
        "EEGNet model test successful."
    )