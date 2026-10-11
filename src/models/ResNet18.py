import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18EEG(nn.Module):
    """
    ResNet-18 adapted for neonatal EEG seizure detection.

    Input:
        [batch_size, 9, 129, 59]

    Output:
        [batch_size, 2]

    Classes:
        0 = Normal
        1 = Seizure
    """

    def __init__(
        self,
        num_channels=9,
        num_classes=2,
        pretrained=True,
        dropout=0.4
    ):
        super().__init__()

        # ---------------------------------------------------------
        # Load ResNet-18
        # ---------------------------------------------------------
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
        else:
            model = resnet18(weights=None)

        # ---------------------------------------------------------
        # Original ResNet-18 expects 3-channel RGB input.
        #
        # Our EEG STFT has 9 channels.
        #
        # Original:
        #     3 -> 64
        #
        # New:
        #     9 -> 64
        # ---------------------------------------------------------

        old_conv = model.conv1

        new_conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        # ---------------------------------------------------------
        # Initialize the new 9-channel convolution using the
        # pretrained 3-channel weights.
        # ---------------------------------------------------------
        if pretrained:
            with torch.no_grad():

                # Repeat RGB filters three times:
                #
                # 3 channels -> 9 channels
                #
                # Divide by 3 to keep the activation magnitude
                # approximately comparable to the original model.
                new_conv.weight.copy_(
                    old_conv.weight.repeat(1, 3, 1, 1) / 3.0
                )

        model.conv1 = new_conv

        # ---------------------------------------------------------
        # Replace ImageNet classifier
        #
        # Original:
        #     512 -> 1000
        #
        # Our task:
        #     512 -> 2
        # ---------------------------------------------------------

        num_features = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(
                num_features,
                num_classes
            )
        )

        self.model = model

    def forward(self, x):
        return self.model(x)


def count_parameters(model):
    """
    Count trainable parameters.
    """
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ==============================================================
# TEST
# ==============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ResNet-18 EEG Model Test")
    print("=" * 60)

    # ----------------------------------------------------------
    # Device
    # ----------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ----------------------------------------------------------
    # Create model
    # ----------------------------------------------------------
    model = ResNet18EEG(
        num_channels=9,
        num_classes=2,
        pretrained=True,
        dropout=0.4
    )

    model = model.to(device)

    print(
        "Trainable parameters:",
        f"{count_parameters(model):,}"
    )

    # ----------------------------------------------------------
    # Test input
    #
    # EXACT SAME SHAPE AS YOUR STFT CACHE
    # ----------------------------------------------------------
    x = torch.randn(
        4,
        9,
        129,
        59,
        device=device
    )

    print("Input shape:", x.shape)

    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------
    model.eval()

    with torch.no_grad():
        output = model(x)

    print("Output shape:", output.shape)

    # ----------------------------------------------------------
    # Convert output to probabilities
    # ----------------------------------------------------------
    probabilities = torch.softmax(
        output,
        dim=1
    )

    print(
        "Probability shape:",
        probabilities.shape
    )

    print(
        "Example probability:",
        probabilities[0].detach().cpu().numpy()
    )

    # ----------------------------------------------------------
    # Test prediction
    # ----------------------------------------------------------
    prediction = torch.argmax(
        probabilities,
        dim=1
    )

    print(
        "Predictions:",
        prediction.detach().cpu().numpy()
    )

    # ----------------------------------------------------------
    # Test backward pass
    # ----------------------------------------------------------
    model.train()

    labels = torch.tensor(
        [0, 1, 0, 1],
        dtype=torch.long,
        device=device
    )

    output = model(x)

    criterion = nn.CrossEntropyLoss()

    loss = criterion(
        output,
        labels
    )

    loss.backward()

    print(
        "Test loss:",
        loss.item()
    )

    print(
        "Backward pass: SUCCESS"
    )

    print("=" * 60)
    print("ResNet-18 test completed successfully.")
    print("=" * 60)