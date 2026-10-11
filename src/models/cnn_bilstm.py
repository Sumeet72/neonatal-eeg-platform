import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):

    def __init__(
        self,
        channels=9,
        classes=2,
        hidden_size=64,
        num_layers=1,
        dropout=0.3
    ):

        super().__init__()

        # ====================================================
        # CNN FEATURE EXTRACTION
        # ====================================================

        self.cnn = nn.Sequential(

            nn.Conv2d(
                channels,
                32,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ELU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Dropout(
                dropout
            ),


            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ELU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Dropout(
                dropout
            )
        )


        # ====================================================
        # ADAPTIVE POOLING
        # ====================================================

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )


        # ====================================================
        # BiLSTM
        # ====================================================

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            )
        )


        # ====================================================
        # CLASSIFIER
        # ====================================================

        self.classifier = nn.Sequential(

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_size * 2,
                classes
            )
        )


    def forward(
        self,
        x
    ):

        # x:
        # [batch, channels, frequency, time]

        x = self.cnn(
            x
        )

        # After CNN:
        # [batch, 64, frequency', time']

        # Treat the remaining time dimension
        # as the sequence dimension.

        x = x.mean(
            dim=2
        )

        # [batch, 64, time']

        x = x.transpose(
            1,
            2
        )

        # [batch, time', 64]

        x, _ = self.lstm(
            x
        )

        # [batch, time', hidden*2]

        x = x[:, -1, :]

        # [batch, hidden*2]

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


    model = CNNBiLSTM(
        channels=9,
        classes=2
    ).to(
        device
    )


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
        "CNN-BiLSTM parameters:",
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
        "CNN-BiLSTM model test successful."
    )