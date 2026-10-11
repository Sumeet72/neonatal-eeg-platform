import torch

from src.models.ResNet18 import ResNet18EEG
from src.utils.device import get_device


class EEGInference:

    def __init__(
        self,
        checkpoint_path="results/final_model/ResNet18_final.pt"
    ):

        self.device = get_device()

        print(
            f"Loading ResNet-18 checkpoint: {checkpoint_path}"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        self.model = ResNet18EEG(
            num_channels=checkpoint["num_channels"],
            num_classes=checkpoint["num_classes"],
            pretrained=False
        ).to(self.device)

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        self.threshold = checkpoint.get(
            "threshold",
            0.40
        )

        print(
            f"Model loaded successfully."
        )

        print(
            f"Device: {self.device}"
        )

        print(
            f"Threshold: {self.threshold}"
        )


    @torch.no_grad()
    def predict(
        self,
        spectrogram
    ):

        if not isinstance(
            spectrogram,
            torch.Tensor
        ):

            spectrogram = torch.tensor(
                spectrogram,
                dtype=torch.float32
            )


        # --------------------------------------------------------
        # Add batch dimension if input is [9, 129, 59]
        # --------------------------------------------------------

        if spectrogram.ndim == 3:

            spectrogram = (
                spectrogram.unsqueeze(0)
            )


        # --------------------------------------------------------
        # Validate input shape
        # --------------------------------------------------------

        if spectrogram.ndim != 4:

            raise ValueError(
                "Expected spectrogram shape "
                "[9, 129, 59] or "
                "[batch, 9, 129, 59]. "
                f"Received: {tuple(spectrogram.shape)}"
            )


        if spectrogram.shape[1:] != (
            9,
            129,
            59
        ):

            raise ValueError(
                "Expected EEG input shape "
                "[batch, 9, 129, 59]. "
                f"Received: {tuple(spectrogram.shape)}"
            )


        spectrogram = spectrogram.to(
            self.device,
            dtype=torch.float32
        )


        # --------------------------------------------------------
        # Model inference
        # --------------------------------------------------------

        outputs = self.model(
            spectrogram
        )


        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        # Class 1 = seizure
        seizure_probability = (
            probabilities[:, 1]
        )


        # --------------------------------------------------------
        # Apply operating threshold
        # --------------------------------------------------------

        predictions = (
            seizure_probability
            >= self.threshold
        ).long()


        # --------------------------------------------------------
        # Format results
        # --------------------------------------------------------

        results = []

        for probability, prediction in zip(
            seizure_probability.cpu().tolist(),
            predictions.cpu().tolist()
        ):

            results.append(
                {
                    "seizure_probability": float(probability),
                    "prediction": int(prediction),
                    "prediction_label": (
                        "seizure"
                        if prediction == 1
                        else "normal"
                    ),
                    "threshold": float(self.threshold)
                }
            )


        return results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Loading EEG inference model..."
    )


    inference = EEGInference()


    # Test with the exact shape used by the
    # cached ResNet-18 model.

    test_input = torch.randn(
        1,
        9,
        129,
        59
    )


    result = inference.predict(
        test_input
    )


    print(
        "\nInference test successful."
    )


    print(
        "Result:",
        result
    )