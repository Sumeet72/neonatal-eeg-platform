import torch


def eeg_to_stft(
    eeg,
    sampling_rate=256,
    n_fft=256,
    hop_length=128,
    win_length=256
):
    """
    Convert EEG from time domain to
    time-frequency representation using STFT.

    Parameters
    ----------
    eeg : torch.Tensor
        Shape:

            (channels, samples)

    sampling_rate : int
        EEG sampling frequency.

    n_fft : int
        FFT size.

    hop_length : int
        Distance between adjacent STFT windows.

    win_length : int
        Length of each STFT window.

    Returns
    -------
    frequencies : torch.Tensor
        Frequency bins.

    times : torch.Tensor
        Time bins.

    power : torch.Tensor
        STFT power.

        Shape:

            (channels, frequencies, time)
    """

    if not isinstance(eeg, torch.Tensor):
        raise TypeError(
            "EEG must be a PyTorch Tensor."
        )

    if eeg.ndim != 2:
        raise ValueError(
            "EEG must have shape "
            "(channels, samples)."
        )

    # Hann window is created directly
    # on the same device as the EEG.
    window = torch.hann_window(
        win_length,
        device=eeg.device,
        dtype=eeg.dtype
    )

    channel_spectrograms = []

    for channel in eeg:

        spectrum = torch.stft(
            channel,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=False,
            return_complex=True
        )

        power = spectrum.abs().pow(2)

        channel_spectrograms.append(
            power
        )

    power = torch.stack(
        channel_spectrograms,
        dim=0
    )

    frequencies = torch.fft.rfftfreq(
        n_fft,
        d=1.0 / sampling_rate,
        device=eeg.device
    )

    times = (
        torch.arange(
            power.shape[-1],
            device=eeg.device
        )
        * hop_length
        / sampling_rate
    )

    return frequencies, times, power


def log_power(power):

    return torch.log10(
        power + 1e-10
    )


def normalize_spectrogram(
    spectrogram,
    epsilon=1e-8
):
    """
    Global z-score normalization of the
    spectrogram tensor.
    """

    mean = spectrogram.mean()

    std = spectrogram.std()

    return (
        spectrogram - mean
    ) / (
        std + epsilon
    )


if __name__ == "__main__":

    from src.utils.device import get_device

    device = get_device()

    sampling_rate = 256

    duration = 30

    samples = (
        sampling_rate * duration
    )

    # Synthetic 9-channel EEG-shaped tensor
    eeg = torch.randn(
        9,
        samples,
        device=device
    )

    print(
        "\nInput:"
    )

    print(
        "Shape:",
        eeg.shape
    )

    print(
        "Device:",
        eeg.device
    )

    frequencies, times, power = eeg_to_stft(
        eeg,
        sampling_rate=sampling_rate
    )

    log_spec = log_power(
        power
    )

    normalized = normalize_spectrogram(
        log_spec
    )

    print(
        "\nSTFT:"
    )

    print(
        "Frequency bins:",
        len(frequencies)
    )

    print(
        "Time bins:",
        len(times)
    )

    print(
        "Power shape:",
        power.shape
    )

    print(
        "Normalized shape:",
        normalized.shape
    )

    print(
        "Power device:",
        power.device
    )

    print(
        "\nGPU STFT test successful."
    )