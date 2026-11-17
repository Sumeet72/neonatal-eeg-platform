from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from .config import settings


# --------------------------------------------------
# DATABASE CONFIGURATION
# --------------------------------------------------

# Backend directory:
# Frontend/backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent

# SQLite database file:
# Frontend/backend/neuroai.db
DATABASE_PATH = BACKEND_DIR / "neuroai.db"

# Use SQLite by default.
# If config.py provides another database URL,
# preserve that configuration.
DATABASE_URL = settings.database_url

connect_args = (
    {"check_same_thread": False}
    if DATABASE_URL.startswith("sqlite")
    else {}
)

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
)


# --------------------------------------------------
# DATABASE SESSION
# --------------------------------------------------

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


# --------------------------------------------------
# BASE CLASS FOR DATABASE MODELS
# --------------------------------------------------

class Base(DeclarativeBase):
    pass


# --------------------------------------------------
# DATABASE DEPENDENCY
# --------------------------------------------------

def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()