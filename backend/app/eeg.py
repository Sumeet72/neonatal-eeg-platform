import csv, io, json, math
from pathlib import Path
import numpy as np
from scipy.signal import stft
from fastapi import HTTPException
from .config import settings

TIME_NAMES = {"time", "timestamp", "seconds", "time_s", "time (s)"}

def parse_eeg_csv(raw: bytes):
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(400, "CSV must be UTF-8 encoded.")
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        raise HTTPException(400, "CSV header row is required.")
    headers = [h.strip() for h in reader.fieldnames]
    time_col = next((h for h in headers if h.lower().strip() in TIME_NAMES), None)
    channels = [h for h in headers if h != time_col]
    rows = []
    for row in reader:
        values = []
        for name in channels:
            raw_value = (row.get(name) or "").strip()
            try:
                value = float(raw_value) if raw_value else None
                if value is not None and not math.isfinite(value): value = None
            except ValueError:
                value = None
            values.append(value)
        if any(v is not None for v in values):
            rows.append(values)
    numeric_channels = []
    matrix_cols = []
    for idx, name in enumerate(channels):
        col = [r[idx] for r in rows]
        if any(v is not None for v in col):
            numeric_channels.append(name)
            matrix_cols.append(col)
    if not numeric_channels or not rows:
        raise HTTPException(400, "No numeric EEG channel columns found.")
    matrix = np.array([[np.nan if v is None else v for v in col] for col in matrix_cols], dtype=float).T
    return numeric_channels, matrix, time_col

def signal_quality(matrix: np.ndarray, channels: list[str]):
    output = []
    for idx, channel in enumerate(channels):
        raw = matrix[:, idx]
        missing = int(np.isnan(raw).sum())
        vals = raw[np.isfinite(raw)]
        issues = []
        if missing:
            issues.append(f"{missing} missing sample(s)")
        if len(vals) < 2:
            issues.append("Insufficient numeric samples")
        else:
            span = float(np.max(vals) - np.min(vals))
            std = float(np.std(vals))
            if span == 0 or std == 0:
                issues.append("Possible flatline")
            if np.max(np.abs(vals)) > 1000:
                issues.append("Large amplitude; verify units")
            if std > 500:
                issues.append("High variability; inspect noise/artifacts")
        status = "POOR" if any("flatline" in x.lower() or "insufficient" in x.lower() for x in issues) else ("WARNING" if issues else "GOOD")
        output.append({"channel": channel, "status": status, "problem": "; ".join(issues) if issues else "No basic rule-based issue detected"})
    return output

def channel_stats(matrix: np.ndarray, channels: list[str], channel: str):
    if channel not in channels:
        raise HTTPException(404, "Channel not found in this recording.")
    x = matrix[:, channels.index(channel)]
    x = x[np.isfinite(x)]
    if len(x) == 0: raise HTTPException(400, "Channel has no usable numeric samples.")
    return {"channel": channel, "sample_count": int(len(x)), "minimum": float(np.min(x)), "maximum": float(np.max(x)), "mean": float(np.mean(x)), "standard_deviation": float(np.std(x)), "rms": float(np.sqrt(np.mean(x*x)))}

def frequency_analysis(matrix: np.ndarray, channels: list[str], channel: str, sampling_rate: float):
    if sampling_rate <= 0: raise HTTPException(400, "Sampling rate must be greater than zero.")
    if channel not in channels: raise HTTPException(404, "Channel not found.")
    x = matrix[:, channels.index(channel)]
    x = np.where(np.isfinite(x), x, np.nan)
    if np.isnan(x).any():
        valid = np.isfinite(x)
        if valid.sum() < 4: raise HTTPException(400, "Not enough valid samples for frequency analysis.")
        x = np.interp(np.arange(len(x)), np.flatnonzero(valid), x[valid])
    if len(x) < 8: raise HTTPException(400, "At least 8 samples are needed for frequency analysis.")
    freqs, times, z = stft(x, fs=sampling_rate, nperseg=min(256, len(x)), noverlap=min(128, max(0, min(256, len(x))//2)))
    power = np.abs(z) ** 2
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, sampling_rate/2)}
    band_power = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        band_power[name] = float(np.mean(power[mask])) if mask.any() else None
    # Limit payload size for UI.
    step_t = max(1, int(math.ceil(len(times)/250)))
    step_f = max(1, int(math.ceil(len(freqs)/100)))
    return {"channel": channel, "sampling_rate": sampling_rate, "frequency_hz": freqs[::step_f].tolist(), "time_seconds": times[::step_t].tolist(), "power": power[::step_f, ::step_t].tolist(), "band_power": band_power, "method": "SciPy STFT; verify parameters against the research model pipeline before clinical/research comparison"}
