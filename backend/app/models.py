from datetime import datetime, timezone

from sqlalchemy import (
    String,
    Integer,
    Float,
    DateTime,
    ForeignKey,
    Text,
    Boolean,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


def now_utc():
    return datetime.now(timezone.utc)


# ==================================================
# USERS
# ==================================================

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    name: Mapped[str] = mapped_column(
        String(120),
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
    )

    password_hash: Mapped[str] = mapped_column(
        String(255),
    )

    role: Mapped[str] = mapped_column(
        String(20),
        default="doctor",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=now_utc,
    )


# ==================================================
# PATIENTS
# ==================================================

class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    # Human-readable patient ID, e.g. 1001, 1002.
    # The backend generates this value.
    patient_id: Mapped[str] = mapped_column(
        String(80),
        unique=True,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(120),
    )

    date_of_birth: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    age: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    gender: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    gestational_age: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    birth_weight: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    birth_status: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    birth_type: Mapped[str | None] = mapped_column(
        String(80),
        nullable=True,
    )

    hospital_unit: Mapped[str | None] = mapped_column(
        String(160),
        nullable=True,
    )

    admission_date: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    clinical_information: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    owner_doctor_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=now_utc,
    )


# ==================================================
# EEG RECORDINGS
# ==================================================

class EEGRecording(Base):
    __tablename__ = "eeg_recordings"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id"),
        index=True,
    )

    filename: Mapped[str] = mapped_column(
        String(255),
    )

    stored_path: Mapped[str] = mapped_column(
        String(500),
    )

    # Original EDF format, for example EDF.
    file_format: Mapped[str] = mapped_column(
        String(20),
        default="EDF",
    )

    # Sampling frequency exposed by the EDF reader.
    sampling_rate: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    # JSON list of channel labels.
    channels_json: Mapped[str] = mapped_column(
        Text,
        default="[]",
    )

    # Number of signals/channels in the recording.
    channel_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    # Number of samples returned by the reader.
    sample_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    duration_seconds: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    file_size_bytes: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    # Upload/validation status; not a seizure diagnosis.
    upload_status: Mapped[str] = mapped_column(
        String(40),
        default="uploaded",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=now_utc,
    )


# ==================================================
# ANALYSIS RESULTS
# ==================================================

class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    recording_id: Mapped[int] = mapped_column(
        ForeignKey("eeg_recordings.id"),
        index=True,
    )

    model_version: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    threshold: Mapped[float] = mapped_column(
        Float,
        default=0.5,
    )

    status: Mapped[str] = mapped_column(
        String(80),
        default="model_not_configured",
    )

    results_json: Mapped[str] = mapped_column(
        Text,
        default="{}",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=now_utc,
    )


# ==================================================
# DOCTOR NOTES
# ==================================================

class DoctorNote(Base):
    __tablename__ = "doctor_notes"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id"),
        index=True,
    )

    recording_id: Mapped[int | None] = mapped_column(
        ForeignKey("eeg_recordings.id"),
        nullable=True,
    )

    doctor_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
    )

    note_type: Mapped[str] = mapped_column(
        String(40),
        default="clinical",
    )

    note: Mapped[str] = mapped_column(
        Text,
    )

    visibility: Mapped[str] = mapped_column(
        String(30),
        default="internal",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=now_utc,
    )


# ==================================================
# PRESCRIPTIONS
# ==================================================

class Prescription(Base):
    __tablename__ = "prescriptions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id"),
        index=True,
    )

    doctor_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
    )

    medicine: Mapped[str] = mapped_column(
        String(200),
    )

    dosage: Mapped[str] = mapped_column(
        String(100),
    )

    frequency: Mapped[str] = mapped_column(
        String(100),
    )

    duration: Mapped[str] = mapped_column(
        String(100),
    )

    instructions: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    confirmed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=now_utc,
    )


# ==================================================
# REPORTS
# ==================================================

class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id"),
        index=True,
    )

    recording_id: Mapped[int] = mapped_column(
        ForeignKey("eeg_recordings.id"),
    )

    doctor_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
    )

    pdf_path: Mapped[str] = mapped_column(
        String(500),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=now_utc,
    )


# ==================================================
# BABY PROFILES
# ==================================================

class BabyProfile(Base):
    __tablename__ = "baby_profiles"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    parent_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        index=True,
    )

    full_name: Mapped[str] = mapped_column(
        String(120),
    )

    date_of_birth: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    birth_time: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    sex: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    blood_group: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True,
    )

    gestational_age_weeks: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    gestational_age_days: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    birth_weight_kg: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    birth_length_cm: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    head_circumference_cm: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    birth_type: Mapped[str | None] = mapped_column(
        String(80),
        nullable=True,
    )

    birth_status: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True,
    )

    hospital_name: Mapped[str | None] = mapped_column(
        String(180),
        nullable=True,
    )

    hospital_unit: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True,
    )

    admission_date: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    discharge_date: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    pediatrician_name: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True,
    )

    pediatrician_phone: Mapped[str | None] = mapped_column(
        String(40),
        nullable=True,
    )

    parent_relationship: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    emergency_contact_name: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True,
    )

    emergency_contact_phone: Mapped[str | None] = mapped_column(
        String(40),
        nullable=True,
    )

    allergies: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    current_medications: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    medical_history: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    pregnancy_birth_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    photo_path: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=now_utc,
    )


# ==================================================
# PARENT-UPLOADED REPORTS
# ==================================================

class ParentUploadedReport(Base):
    __tablename__ = "parent_uploaded_reports"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    parent_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        index=True,
    )

    baby_id: Mapped[int | None] = mapped_column(
        ForeignKey("baby_profiles.id"),
        nullable=True,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(180),
    )

    original_filename: Mapped[str] = mapped_column(
        String(255),
    )

    stored_path: Mapped[str] = mapped_column(
        String(500),
    )

    content_type: Mapped[str] = mapped_column(
        String(100),
    )

    file_size: Mapped[int] = mapped_column(
        Integer,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=now_utc,
    )