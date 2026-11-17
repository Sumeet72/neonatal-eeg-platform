from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Backend directory:
# Frontend/backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):

    app_name: str = "NeuroAI - Neonatal EEG Platform"

    # SQLite database
    database_url: str = (
        f"sqlite:///{(BACKEND_DIR / 'neuroai.db').as_posix()}"
    )

    # Authentication
    jwt_secret: str = "replace-this-with-a-long-random-secret"
    jwt_expire_minutes: int = 60

    # File storage
    upload_dir: str = str(BACKEND_DIR / "uploads")
    report_dir: str = str(BACKEND_DIR / "reports")

    # Maximum upload size
    max_upload_size_mb: int = 100

    # Demo account creation
    seed_demo_user: bool = False

    # React frontend
    frontend_origins: str = "http://localhost:5173"

    # Trained model checkpoint
    model_path: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()