"""Pure factorized spatio-temporal Transformer for neonatal EEG STFT epochs.

Input:  [batch, 9 EEG channels, 129 frequency bins, 59 time frames]
Output: [batch, 2 class logits] where class 0=normal and class 1=seizure.

This model intentionally contains no convolutional layers.
"""

import torch
import torch.nn as nn


class EEGSpatioTemporalTransformer(nn.Module):
    """Factorized temporal-then-channel Transformer classifier.

    For each channel, the 129-bin spectrum at each STFT time frame is
    projected into an embedding. Self-attention is then applied across time
    independently for each channel, followed by self-attention across
    channels independently at each time frame.
    """

    def __init__(
        self,
        num_channels: int = 9,
        num_frequency_bins: int = 129,
        num_time_frames: int = 59,
        num_classes: int = 2,
        embed_dim: int = 64,
        num_heads: int = 4,
        temporal_layers: int = 2,
        channel_layers: int = 1,
        feedforward_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if min(num_channels, num_frequency_bins, num_time_frames) <= 0:
            raise ValueError("Input dimensions must be positive.")
        if temporal_layers < 1 or channel_layers < 1:
            raise ValueError("At least one temporal and channel layer is required.")

        self.num_channels = num_channels
        self.num_frequency_bins = num_frequency_bins
        self.num_time_frames = num_time_frames
        self.embed_dim = embed_dim

        # Each channel/time token is the full log-power spectrum at that time.
        self.spectrum_norm = nn.LayerNorm(num_frequency_bins)
        self.spectrum_projection = nn.Linear(num_frequency_bins, embed_dim)

        # Learnable temporal and channel identity embeddings.
        self.temporal_position = nn.Parameter(
            torch.zeros(1, 1, num_time_frames, embed_dim)
        )
        self.channel_embedding = nn.Parameter(
            torch.zeros(1, num_channels, 1, embed_dim)
        )
        self.input_dropout = nn.Dropout(dropout)

        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_encoder_layer,
            num_layers=temporal_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        channel_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.channel_encoder = nn.TransformerEncoder(
            channel_encoder_layer,
            num_layers=channel_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        self.output_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        nn.init.trunc_normal_(self.temporal_position, std=0.02)
        nn.init.trunc_normal_(self.channel_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return two-class logits for an STFT batch.

        Args:
            x: Tensor shaped [B, C, F, T], expected [B, 9, 129, 59].

        Returns:
            Logits shaped [B, 2]. Apply softmax(dim=1) for probabilities.
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected a 4-D [B,C,F,T] tensor, received shape {tuple(x.shape)}."
            )

        expected = (
            self.num_channels,
            self.num_frequency_bins,
            self.num_time_frames,
        )
        actual = tuple(x.shape[1:])
        if actual != expected:
            raise ValueError(
                f"Expected input shape [B,{expected[0]},{expected[1]},{expected[2]}], "
                f"received [B,{actual[0]},{actual[1]},{actual[2]}]."
            )

        if not torch.is_floating_point(x):
            x = x.float()
        if not torch.isfinite(x).all():
            raise ValueError("Input STFT contains NaN or infinite values.")

        # [B,C,F,T] -> [B,C,T,F] -> [B,C,T,D]
        x = x.permute(0, 1, 3, 2).contiguous()
        x = self.spectrum_norm(x)
        x = self.spectrum_projection(x)
        x = x + self.temporal_position + self.channel_embedding
        x = self.input_dropout(x)

        batch_size, channels, frames, dim = x.shape

        # Temporal attention: every channel attends over the 59 time frames.
        x = x.reshape(batch_size * channels, frames, dim)
        x = self.temporal_encoder(x)
        x = x.reshape(batch_size, channels, frames, dim)

        # Channel attention: at every time frame, channels attend to each other.
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(batch_size * frames, channels, dim)
        x = self.channel_encoder(x)
        x = x.reshape(batch_size, frames, channels, dim)
        x = x.permute(0, 2, 1, 3).contiguous()

        # Global mean pooling over channels and time; no CNN is used.
        x = self.output_norm(x)
        x = x.mean(dim=(1, 2))
        return self.classifier(x)


def build_eeg_transformer(**kwargs) -> EEGSpatioTemporalTransformer:
    """Factory function for the default pure Transformer model."""
    return EEGSpatioTemporalTransformer(**kwargs)


if __name__ == "__main__":
    from src.utils.device import get_device

    device = get_device()
    model = EEGSpatioTemporalTransformer().to(device)
    inputs = torch.randn(2, 9, 129, 59, device=device)
    logits = model(inputs)
    probabilities = torch.softmax(logits, dim=1)
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1], device=device))
    loss.backward()

    print("Pure EEG Spatio-Temporal Transformer smoke test")
    print("Device:", device)
    print("Input shape:", tuple(inputs.shape))
    print("Logit shape:", tuple(logits.shape))
    print("Probability shape:", tuple(probabilities.shape))
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Loss finite:", bool(torch.isfinite(loss).item()))
    print("Backward pass: SUCCESS")
