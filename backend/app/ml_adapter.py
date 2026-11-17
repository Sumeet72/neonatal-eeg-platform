from pathlib import Path
from .config import settings

def model_status():
    configured = bool(settings.model_path and Path(settings.model_path).exists())
    return {
        "status": "model_configured_not_loaded" if configured else "model_not_configured",
        "available": False,
        "message": "The Nathan + Sumit model integration and validated checkpoint are not included. No seizure predictions, confidence, events, or localization have been generated.",
        "model_path_configured": bool(settings.model_path),
    }

def run_model(*args, **kwargs):
    # Integrate the actual scientific pipeline here. Do not replace with synthetic predictions.
    return model_status()
