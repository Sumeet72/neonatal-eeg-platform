import numpy as np
from app.eeg import signal_quality

def test_signal_quality_detects_flatline():
    result = signal_quality(np.ones((20, 1)), ["Fp1"])
    assert result[0]["status"] == "POOR"
    assert "flatline" in result[0]["problem"].lower()

def test_signal_quality_marks_missing_values():
    data = np.array([[1.0], [np.nan], [2.0]])
    result = signal_quality(data, ["C3"])
    assert result[0]["status"] == "WARNING"
