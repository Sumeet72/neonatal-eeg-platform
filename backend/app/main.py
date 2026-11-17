import csv, json, os, shutil
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from pydantic import BaseModel, Field, field_validator
from .config import settings
from .database import Base, engine, get_db, SessionLocal
from .models import User, Patient, EEGRecording, AnalysisResult, DoctorNote, Prescription, Report, BabyProfile, ParentUploadedReport
from .schemas import LoginInput, PatientCreate, NoteCreate, PrescriptionCreate
from .security import hash_password, verify_password, create_access_token, current_user, require_doctor
from .eeg import parse_eeg_csv, signal_quality, channel_stats, frequency_analysis
import mne
from .ml_adapter import run_model, model_status
from .reports import make_report

class ParentRegisterInput(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Name is required.")
        return value

    @field_validator("email")
    @classmethod
    def clean_email(cls, value: str) -> str:
        value = value.strip().lower()
        if "@" not in value or "." not in value.rsplit("@", 1)[-1]:
            raise ValueError("Enter a valid email address.")
        return value


app = FastAPI(title=settings.app_name, version="0.1.0")
origins = [x.strip() for x in settings.frontend_origins.split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
UPLOAD_DIR = Path(settings.upload_dir); REPORT_DIR = Path(settings.report_dir)
PARENT_BABY_PHOTO_DIR = Path(settings.upload_dir) / "baby_photos"
PARENT_REPORT_DIR = Path(settings.upload_dir) / "parent_reports"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True); REPORT_DIR.mkdir(parents=True, exist_ok=True); PARENT_BABY_PHOTO_DIR.mkdir(parents=True, exist_ok=True); PARENT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
    if settings.seed_demo_user:
        db = SessionLocal()
        try:
            if not db.scalar(select(User).where(User.email == "doctor@example.com")):
                db.add(User(name="Demo clinician", email="doctor@example.com", password_hash=hash_password("ChangeMe123!"), role="doctor"))
                db.commit()
        finally:
            db.close()

@app.get("/api/health")
def health():
    return {"status": "ok", "model": model_status()}

@app.post("/api/auth/register", status_code=status.HTTP_201_CREATED)
def register_parent(payload: ParentRegisterInput, db: Session = Depends(get_db)):
    """Create a parent account. Public registration never creates doctor accounts."""
    email = payload.email.strip().lower()
    existing = db.scalar(select(User).where(User.email == email))
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists. Please log in instead.")

    user = User(
        name=payload.name.strip(),
        email=email,
        password_hash=hash_password(payload.password),
        role="patient",
    )
    db.add(user)
    try:
        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="An account with this email already exists. Please log in instead.")

    return {
        "message": "Parent account created successfully. You can now log in.",
        "user": {"id": user.id, "name": user.name, "email": user.email, "role": user.role},
    }


@app.post("/api/auth/login")
def login(payload: LoginInput, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == payload.email.lower()))
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(401, "Incorrect email or password")
    return {"access_token": create_access_token(user), "token_type": "bearer", "user": {"id": user.id, "name": user.name, "email": user.email, "role": user.role}}

@app.post("/api/auth/logout")
def logout(user: User = Depends(current_user)):
    # Stateless access token expires naturally; revoke/refresh-token rotation belongs in production auth.
    return {"message": "Logged out on client. Access token remains valid until expiry in this starter."}

@app.get("/api/me")
def me(user: User = Depends(current_user)):
    return {"id": user.id, "name": user.name, "email": user.email, "role": user.role}



def baby_profile_dict(baby: BabyProfile):
    fields = ["id", "full_name", "date_of_birth", "birth_time", "sex", "blood_group",
              "gestational_age_weeks", "gestational_age_days", "birth_weight_kg",
              "birth_length_cm", "head_circumference_cm", "birth_type", "birth_status",
              "hospital_name", "hospital_unit", "admission_date", "discharge_date",
              "pediatrician_name", "pediatrician_phone", "parent_relationship",
              "emergency_contact_name", "emergency_contact_phone", "allergies",
              "current_medications", "medical_history", "pregnancy_birth_notes", "created_at"]
    result = {field: getattr(baby, field) for field in fields}
    result["photo_url"] = f"/api/parent/babies/{baby.id}/photo" if baby.photo_path else None
    result["information_notice"] = "Parent-entered information; not verified by a clinician."
    return result

@app.get("/api/parent/babies")
def list_parent_babies(user: User = Depends(current_user), db: Session = Depends(get_db)):
    if user.role != "patient":
        raise HTTPException(403, "This endpoint is for Parent Mode accounts.")
    babies = db.scalars(select(BabyProfile).where(BabyProfile.parent_id == user.id).order_by(BabyProfile.created_at.desc())).all()
    return [baby_profile_dict(baby) for baby in babies]

@app.post("/api/parent/babies", status_code=201)
async def create_parent_baby(
    full_name: str = Form(...), date_of_birth: str | None = Form(None), birth_time: str | None = Form(None),
    sex: str | None = Form(None), blood_group: str | None = Form(None),
    gestational_age_weeks: int | None = Form(None), gestational_age_days: int | None = Form(None),
    birth_weight_kg: float | None = Form(None), birth_length_cm: float | None = Form(None),
    head_circumference_cm: float | None = Form(None), birth_type: str | None = Form(None),
    birth_status: str | None = Form(None), hospital_name: str | None = Form(None), hospital_unit: str | None = Form(None),
    admission_date: str | None = Form(None), discharge_date: str | None = Form(None),
    pediatrician_name: str | None = Form(None), pediatrician_phone: str | None = Form(None),
    parent_relationship: str | None = Form(None), emergency_contact_name: str | None = Form(None),
    emergency_contact_phone: str | None = Form(None), allergies: str | None = Form(None),
    current_medications: str | None = Form(None), medical_history: str | None = Form(None),
    pregnancy_birth_notes: str | None = Form(None), photo: UploadFile | None = File(None),
    user: User = Depends(current_user), db: Session = Depends(get_db)):
    if user.role != "patient":
        raise HTTPException(403, "Only Parent Mode accounts can create a family baby profile.")
    name = full_name.strip()
    if not name:
        raise HTTPException(422, "Baby's full name is required.")
    photo_path = None
    if photo and photo.filename:
        if photo.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(400, "Photo must be JPG, PNG, or WEBP.")
        content = await photo.read()
        if len(content) > 5 * 1024 * 1024:
            raise HTTPException(413, "Photo must be 5 MB or smaller.")
        # Check file signatures instead of trusting the extension alone.
        valid = (content.startswith(b"\xff\xd8\xff") if photo.content_type == "image/jpeg" else
                 content.startswith(b"\x89PNG\r\n\x1a\n") if photo.content_type == "image/png" else
                 content.startswith(b"RIFF") and content[8:12] == b"WEBP")
        if not valid:
            raise HTTPException(400, "The selected file is not a valid JPG, PNG, or WEBP image.")
        ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[photo.content_type]
        destination = PARENT_BABY_PHOTO_DIR / f"parent{user.id}_{os.urandom(12).hex()}{ext}"
        destination.write_bytes(content)
        photo_path = str(destination.resolve())
    fields = {"full_name": name, "date_of_birth": date_of_birth, "birth_time": birth_time, "sex": sex,
      "blood_group": blood_group, "gestational_age_weeks": gestational_age_weeks, "gestational_age_days": gestational_age_days,
      "birth_weight_kg": birth_weight_kg, "birth_length_cm": birth_length_cm, "head_circumference_cm": head_circumference_cm,
      "birth_type": birth_type, "birth_status": birth_status, "hospital_name": hospital_name, "hospital_unit": hospital_unit,
      "admission_date": admission_date, "discharge_date": discharge_date, "pediatrician_name": pediatrician_name,
      "pediatrician_phone": pediatrician_phone, "parent_relationship": parent_relationship,
      "emergency_contact_name": emergency_contact_name, "emergency_contact_phone": emergency_contact_phone,
      "allergies": allergies, "current_medications": current_medications, "medical_history": medical_history,
      "pregnancy_birth_notes": pregnancy_birth_notes, "photo_path": photo_path}
    baby = BabyProfile(parent_id=user.id, **fields)
    db.add(baby)
    db.commit()
    db.refresh(baby)
    return baby_profile_dict(baby)

@app.get("/api/parent/babies/{baby_id}/photo")
def get_parent_baby_photo(baby_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if user.role != "patient":
        raise HTTPException(403, "This endpoint is for Parent Mode accounts.")
    baby = db.get(BabyProfile, baby_id)
    if not baby or baby.parent_id != user.id:
        raise HTTPException(404, "Baby profile or photo not found.")
    if not baby.photo_path or not Path(baby.photo_path).is_file():
        raise HTTPException(404, "No photo has been uploaded for this baby.")
    return FileResponse(baby.photo_path)

@app.get("/api/parent/reports")
def list_parent_reports(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """List only documents uploaded by the authenticated parent account."""
    if user.role != "patient":
        raise HTTPException(status_code=403, detail="This endpoint is for Parent Mode accounts.")
    rows = db.scalars(select(ParentUploadedReport).where(ParentUploadedReport.parent_id == user.id).order_by(ParentUploadedReport.created_at.desc())).all()
    return [{"id": row.id, "title": row.title, "original_filename": row.original_filename, "content_type": row.content_type, "file_size": row.file_size, "notes": row.notes, "baby_id": row.baby_id, "created_at": row.created_at, "download_url": f"/api/parent/reports/{row.id}/download", "document_type": "parent_uploaded_document", "clinical_status": "Not verified by a clinician"} for row in rows]


@app.post("/api/parent/reports", status_code=status.HTTP_201_CREATED)
async def upload_parent_report(
    file: UploadFile = File(...), title: str = Form(""), notes: str | None = Form(None),
    baby_id: int | None = Form(None), user: User = Depends(current_user), db: Session = Depends(get_db)
):
    """Upload a report document for the parent's own records. This does not run EEG analysis."""
    if user.role != "patient":
        raise HTTPException(status_code=403, detail="Only Parent Mode accounts can upload family documents.")
    if baby_id is not None:
        baby = db.get(BabyProfile, baby_id)
        if not baby or baby.parent_id != user.id:
            raise HTTPException(status_code=404, detail="Baby profile not found for this account.")
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Choose a file to upload.")
    allowed = {"application/pdf": ".pdf", "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
    content_type = (file.content_type or "").lower()
    if content_type not in allowed:
        raise HTTPException(status_code=400, detail="Upload a PDF, JPG, PNG, or WEBP report.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The selected file is empty.")
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Report files must be 10 MB or smaller.")
    valid = (content.startswith(b"%PDF-") if content_type == "application/pdf" else
             content.startswith(b"\xff\xd8\xff") if content_type == "image/jpeg" else
             content.startswith(b"\x89PNG\r\n\x1a\n") if content_type == "image/png" else
             content.startswith(b"RIFF") and content[8:12] == b"WEBP")
    if not valid:
        raise HTTPException(status_code=400, detail="The file contents do not match the selected file type.")
    stored_name = f"parent{user.id}_{os.urandom(12).hex()}{allowed[content_type]}"
    destination = PARENT_REPORT_DIR / stored_name
    destination.write_bytes(content)
    safe_title = title.strip()[:180] or filename
    row = ParentUploadedReport(parent_id=user.id, baby_id=baby_id, title=safe_title, original_filename=filename[:255],
                               stored_path=str(destination.resolve()), content_type=content_type, file_size=len(content), notes=(notes or "").strip() or None)
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"id": row.id, "title": row.title, "original_filename": row.original_filename, "content_type": row.content_type, "file_size": row.file_size, "notes": row.notes, "baby_id": row.baby_id, "created_at": row.created_at, "download_url": f"/api/parent/reports/{row.id}/download", "document_type": "parent_uploaded_document", "clinical_status": "Not verified by a clinician"}


@app.get("/api/parent/reports/{report_id}/download")
def download_parent_report(report_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if user.role != "patient":
        raise HTTPException(status_code=403, detail="This endpoint is for Parent Mode accounts.")
    row = db.get(ParentUploadedReport, report_id)
    if not row or row.parent_id != user.id:
        raise HTTPException(status_code=404, detail="Uploaded report not found.")
    path = Path(row.stored_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="The stored report file is missing.")
    return FileResponse(path, media_type=row.content_type, filename=row.original_filename, content_disposition_type="attachment")


@app.get("/api/dashboard")
def dashboard(user: User = Depends(current_user), db: Session = Depends(get_db)):
    if user.role == "patient":
        return {"role": "patient", "message": "Patient dashboard data is scoped to the authenticated patient account."}
    patients = db.scalars(select(Patient).where(Patient.owner_doctor_id == user.id)).all()
    patient_ids = [p.id for p in patients]
    recordings = db.scalars(select(EEGRecording).where(EEGRecording.patient_id.in_(patient_ids))).all() if patient_ids else []
    analyses = db.scalars(select(AnalysisResult).where(AnalysisResult.recording_id.in_([r.id for r in recordings]))).all() if recordings else []
    return {"role": "doctor", "total_patients": len(patients), "total_recordings": len(recordings),
            "analyses_completed": sum(1 for a in analyses if a.status == "completed"),
            "detected_seizure_events": None, "seizure_events_note": "Unavailable until validated model output is integrated",
            "recent_patients": [{"id": p.id, "patient_id": p.patient_id, "name": p.name, "age": p.age, "gestational_age": p.gestational_age} for p in patients[-8:]]}

@app.get("/api/patients")
def list_patients(user: User = Depends(current_user), db: Session = Depends(get_db)):
    if user.role == "patient":
        raise HTTPException(403, "Patient accounts can only access their own assigned record; patient-account linking is not configured in this starter.")
    rows = db.scalars(select(Patient).where(Patient.owner_doctor_id == user.id).order_by(Patient.created_at.desc())).all()
    return [patient_dict(p) for p in rows]

def patient_dict(p):
    return {k: getattr(p, k) for k in ["id","patient_id","name","date_of_birth","age","gender","gestational_age","birth_weight","birth_status","birth_type","hospital_unit","admission_date","clinical_information","owner_doctor_id","created_at"]}

@app.post("/api/patients")
def create_patient(payload: PatientCreate, user: User = Depends(require_doctor), db: Session = Depends(get_db)):
    if db.scalar(select(Patient).where(Patient.patient_id == payload.patient_id)):
        raise HTTPException(409, "Patient ID must be unique.")
    p = Patient(**payload.model_dump(), owner_doctor_id=user.id)
    db.add(p); db.commit(); db.refresh(p)
    return patient_dict(p)

@app.get("/api/patients/{patient_id}")
def get_patient(patient_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    p = db.get(Patient, patient_id)
    if not p: raise HTTPException(404, "Patient not found")
    if user.role != "doctor" or p.owner_doctor_id != user.id: raise HTTPException(403, "Not authorized for this patient")
    return patient_dict(p)

@app.get("/api/patients/{patient_id}/history")
def patient_history(patient_id: int, user: User = Depends(require_doctor), db: Session = Depends(get_db)):
    p = db.get(Patient, patient_id)
    if not p or p.owner_doctor_id != user.id: raise HTTPException(404, "Patient not found")
    recs = db.scalars(select(EEGRecording).where(EEGRecording.patient_id == p.id).order_by(EEGRecording.created_at.desc())).all()
    notes = db.scalars(select(DoctorNote).where(DoctorNote.patient_id == p.id).order_by(DoctorNote.created_at.desc())).all()
    rx = db.scalars(select(Prescription).where(Prescription.patient_id == p.id).order_by(Prescription.created_at.desc())).all()
    reports = db.scalars(select(Report).where(Report.patient_id == p.id).order_by(Report.created_at.desc())).all()
    return {"patient": patient_dict(p), "recordings": [recording_dict(r) for r in recs],
            "notes": [{"id": n.id, "note": n.note, "visibility": n.visibility, "note_type": n.note_type, "created_at": n.created_at} for n in notes],
            "prescriptions": [{"id": x.id, "medicine": x.medicine, "dosage": x.dosage, "frequency": x.frequency, "duration": x.duration, "instructions": x.instructions, "confirmed": x.confirmed} for x in rx],
            "reports": [{"id": r.id, "recording_id": r.recording_id, "created_at": r.created_at} for r in reports]}

def recording_dict(r):
    return {"id": r.id, "patient_id": r.patient_id, "filename": r.filename, "sampling_rate": r.sampling_rate,
            "channels": json.loads(r.channels_json), "sample_count": r.sample_count, "duration_seconds": r.duration_seconds, "created_at": r.created_at}

def load_matrix(recording: EEGRecording):
    """Load a stored CSV/TSV or EDF recording for basic signal analysis."""
    path = Path(recording.stored_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="The stored EEG recording file could not be found.")

    extension = path.suffix.lower()
    if extension == ".edf":
        raw = None
        try:
            raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
            channels = list(raw.ch_names)
            matrix = raw.get_data().T  # (samples, channels)
            if matrix.size == 0 or not channels:
                raise HTTPException(status_code=400, detail="The EDF file contains no usable signals.")
            return channels, matrix
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to read EDF recording: {exc}")
        finally:
            if raw is not None:
                raw.close()

    if extension in {".csv", ".tsv"}:
        channels, matrix, _ = parse_eeg_csv(path.read_bytes())
        return channels, matrix

    raise HTTPException(status_code=400, detail="Unsupported recording format. Use CSV, TSV, or EDF.")

@app.post("/api/eeg/upload")
async def upload_eeg(
    patient_id: int = Form(...),
    sampling_rate: float | None = Form(None),
    file: UploadFile = File(...),
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    patient = db.get(Patient, patient_id)
    if not patient or patient.owner_doctor_id != user.id:
        raise HTTPException(status_code=404, detail="Authorized patient not found.")

    original_name = Path(file.filename or "").name
    if not original_name:
        raise HTTPException(status_code=400, detail="Please select a file to upload.")

    extension = Path(original_name).suffix.lower()
    if extension not in {".csv", ".tsv", ".edf"}:
        raise HTTPException(status_code=400, detail="Unsupported file format. Upload a CSV, TSV, or EDF file.")

    content = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if not content:
        raise HTTPException(status_code=400, detail="The selected file is empty.")
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds the {settings.max_upload_size_mb} MB upload limit.")

    # A unique server-side filename prevents overwriting an earlier upload.
    destination = UPLOAD_DIR / f"patient{patient.id}_{os.urandom(8).hex()}{extension}"
    destination.write_bytes(content)

    try:
        if extension == ".edf":
            raw = None
            try:
                raw = mne.io.read_raw_edf(str(destination), preload=True, verbose="ERROR")
                channels = list(raw.ch_names)
                matrix = raw.get_data().T
                detected_rate = float(raw.info["sfreq"])
                sample_count = int(raw.n_times)
                duration = sample_count / detected_rate if detected_rate > 0 else None
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"The EDF file could not be read: {exc}")
            finally:
                if raw is not None:
                    raw.close()

            if not channels or matrix.size == 0:
                raise HTTPException(status_code=400, detail="The EDF file contains no usable signals.")
            stored_sampling_rate = detected_rate
            time_col_detected = False
        else:
            channels, matrix, time_col = parse_eeg_csv(content)
            if sampling_rate is not None and sampling_rate <= 0:
                raise HTTPException(status_code=400, detail="Sampling rate must be greater than zero.")
            stored_sampling_rate = sampling_rate
            sample_count = int(matrix.shape[0])
            duration = sample_count / sampling_rate if sampling_rate else None
            time_col_detected = bool(time_col)

        rec = EEGRecording(
            patient_id=patient.id,
            filename=original_name,
            stored_path=str(destination.resolve()),
            sampling_rate=stored_sampling_rate,
            channels_json=json.dumps(channels),
            sample_count=sample_count,
            duration_seconds=duration,
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)

        return {
            "recording": recording_dict(rec),
            "upload_status": "validated",
            "time_column_detected": time_col_detected,
            "signal_quality": signal_quality(matrix, channels),
            "message": "Recording uploaded and basic signal checks completed. No seizure-detection model has been run.",
        }
    except Exception:
        db.rollback()
        if destination.exists():
            destination.unlink()
        raise

@app.get("/api/eeg/{recording_id}")
def get_recording(recording_id: int, user: User = Depends(require_doctor), db: Session = Depends(get_db)):
    r = db.get(EEGRecording, recording_id)
    if not r: raise HTTPException(404, "Recording not found")
    p = db.get(Patient, r.patient_id)
    if not p or p.owner_doctor_id != user.id: raise HTTPException(403, "Not authorized")
    channels, matrix = load_matrix(r)
    return {**recording_dict(r), "signal_quality": signal_quality(matrix, channels)}

@app.get("/api/analysis/{recording_id}")
def get_analysis(recording_id: int, user: User = Depends(require_doctor), db: Session = Depends(get_db)):
    r = db.get(EEGRecording, recording_id)
    if not r: raise HTTPException(404, "Recording not found")
    p = db.get(Patient, r.patient_id)
    if not p or p.owner_doctor_id != user.id: raise HTTPException(403, "Not authorized")
    result = db.scalar(select(AnalysisResult).where(AnalysisResult.recording_id == r.id).order_by(AnalysisResult.created_at.desc()))
    return {"recording": recording_dict(r), "analysis": json.loads(result.results_json) if result else run_model(),
            "status": result.status if result else model_status()["status"], "events": [], "events_note": "Not available from current model."}

@app.post("/api/analysis/{recording_id}/run")
def run_analysis(recording_id: int, user: User = Depends(require_doctor), db: Session = Depends(get_db)):
    r = db.get(EEGRecording, recording_id)
    if not r: raise HTTPException(404, "Recording not found")
    p = db.get(Patient, r.patient_id)
    if not p or p.owner_doctor_id != user.id: raise HTTPException(403, "Not authorized")
    output = run_model()
    result = AnalysisResult(recording_id=r.id, threshold=0.5, status=output["status"], results_json=json.dumps(output))
    db.add(result); db.commit()
    return {"analysis_id": result.id, **output}

@app.get("/api/analysis/{recording_id}/channels")
def channel_list(recording_id: int, user: User = Depends(require_doctor), db: Session = Depends(get_db)):
    r = db.get(EEGRecording, recording_id)
    if not r: raise HTTPException(404, "Recording not found")
    p = db.get(Patient, r.patient_id)
    if not p or p.owner_doctor_id != user.id: raise HTTPException(403, "Not authorized")
    channels, matrix = load_matrix(r)
    return {"channels": channels, "signal_quality": signal_quality(matrix, channels)}

@app.get("/api/analysis/{recording_id}/channels/{channel}")
def get_channel(recording_id: int, channel: str, user: User = Depends(require_doctor), db: Session = Depends(get_db)):
    r = db.get(EEGRecording, recording_id)
    if not r: raise HTTPException(404, "Recording not found")
    p = db.get(Patient, r.patient_id)
    if not p or p.owner_doctor_id != user.id: raise HTTPException(403, "Not authorized")
    channels, matrix = load_matrix(r)
    return channel_stats(matrix, channels, channel)

@app.get("/api/analysis/{recording_id}/spectrogram")
def get_spectrogram(recording_id: int, channel: str, sampling_rate: float | None = None,
                    user: User = Depends(require_doctor), db: Session = Depends(get_db)):
    r = db.get(EEGRecording, recording_id)
    if not r: raise HTTPException(404, "Recording not found")
    p = db.get(Patient, r.patient_id)
    if not p or p.owner_doctor_id != user.id: raise HTTPException(403, "Not authorized")
    fs = sampling_rate or r.sampling_rate
    if not fs: raise HTTPException(400, "Sampling rate is required for frequency analysis.")
    channels, matrix = load_matrix(r)
    return frequency_analysis(matrix, channels, channel, fs)

@app.get("/api/analysis/{recording_id}/quality")
def get_quality(recording_id: int, user: User = Depends(require_doctor), db: Session = Depends(get_db)):
    r = db.get(EEGRecording, recording_id)
    if not r: raise HTTPException(404, "Recording not found")
    p = db.get(Patient, r.patient_id)
    if not p or p.owner_doctor_id != user.id: raise HTTPException(403, "Not authorized")
    channels, matrix = load_matrix(r)
    return {"channels": signal_quality(matrix, channels), "method": "Basic rule-based checks only; not a clinical impedance measurement"}

@app.post("/api/doctor-notes")
def create_note(payload: NoteCreate, user: User = Depends(require_doctor), db: Session = Depends(get_db)):
    p = db.get(Patient, payload.patient_id)
    if not p or p.owner_doctor_id != user.id: raise HTTPException(404, "Patient not found")
    if payload.visibility not in {"internal", "patient"}: raise HTTPException(400, "Visibility must be internal or patient")
    note = DoctorNote(**payload.model_dump(), doctor_id=user.id)
    db.add(note); db.commit(); db.refresh(note)
    return {"id": note.id, "patient_id": note.patient_id, "visibility": note.visibility, "note": note.note, "created_at": note.created_at}

@app.get("/api/doctor-notes/{patient_id}")
def list_notes(patient_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    p = db.get(Patient, patient_id)
    if not p: raise HTTPException(404, "Patient not found")
    if user.role != "doctor" or p.owner_doctor_id != user.id: raise HTTPException(403, "Not authorized")
    rows = db.scalars(select(DoctorNote).where(DoctorNote.patient_id == patient_id).order_by(DoctorNote.created_at.desc())).all()
    return [{"id": n.id, "note": n.note, "visibility": n.visibility, "note_type": n.note_type, "created_at": n.created_at} for n in rows]

@app.post("/api/prescriptions")
def create_prescription(payload: PrescriptionCreate, user: User = Depends(require_doctor), db: Session = Depends(get_db)):
    if not payload.confirmed: raise HTTPException(400, "Clinician confirmation is required before saving a prescription.")
    p = db.get(Patient, payload.patient_id)
    if not p or p.owner_doctor_id != user.id: raise HTTPException(404, "Patient not found")
    rx = Prescription(**payload.model_dump(), doctor_id=user.id)
    db.add(rx); db.commit(); db.refresh(rx)
    return {"id": rx.id, "patient_id": rx.patient_id, "medicine": rx.medicine, "confirmed": rx.confirmed}

@app.get("/api/prescriptions/{patient_id}")
def list_prescriptions(patient_id: int, user: User = Depends(require_doctor), db: Session = Depends(get_db)):
    p = db.get(Patient, patient_id)
    if not p or p.owner_doctor_id != user.id: raise HTTPException(404, "Patient not found")
    rows = db.scalars(select(Prescription).where(Prescription.patient_id == patient_id)).all()
    return [{"id": x.id, "medicine": x.medicine, "dosage": x.dosage, "frequency": x.frequency, "duration": x.duration, "instructions": x.instructions, "confirmed": x.confirmed} for x in rows]

@app.post("/api/reports/generate")
def generate_report(recording_id: int, user: User = Depends(require_doctor), db: Session = Depends(get_db)):
    r = db.get(EEGRecording, recording_id)
    if not r: raise HTTPException(404, "Recording not found")
    p = db.get(Patient, r.patient_id)
    if not p or p.owner_doctor_id != user.id: raise HTTPException(403, "Not authorized")
    status = model_status()["status"]
    path = (REPORT_DIR / f"report_{p.id}_{r.id}.pdf").resolve()
    make_report(str(path), patient_dict(p), {**recording_dict(r), "channels": json.loads(r.channels_json)}, status, user.name)
    report = Report(patient_id=p.id, recording_id=r.id, doctor_id=user.id, pdf_path=str(path))
    db.add(report); db.commit(); db.refresh(report)
    return {"report_id": report.id, "download_url": f"/api/reports/{report.id}/download", "status": "generated_with_unavailable_model_outputs"}

@app.get("/api/reports")
def list_reports(user: User = Depends(require_doctor), db: Session = Depends(get_db)):
    patient_ids = [p.id for p in db.scalars(select(Patient).where(Patient.owner_doctor_id == user.id)).all()]
    if not patient_ids: return []
    rows = db.scalars(select(Report).where(Report.patient_id.in_(patient_ids)).order_by(Report.created_at.desc())).all()
    return [{"id": r.id, "patient_id": r.patient_id, "recording_id": r.recording_id, "created_at": r.created_at} for r in rows]

@app.get("/api/reports/{report_id}/download")
def download_report(report_id: int, user: User = Depends(require_doctor), db: Session = Depends(get_db)):
    r = db.get(Report, report_id)
    if not r: raise HTTPException(404, "Report not found")
    p = db.get(Patient, r.patient_id)
    if not p or p.owner_doctor_id != user.id: raise HTTPException(403, "Not authorized")
    if not Path(r.pdf_path).exists(): raise HTTPException(404, "PDF file missing")
    return FileResponse(r.pdf_path, media_type="application/pdf", filename=f"neonatal_eeg_report_{report_id}.pdf")

@app.get("/api/analysis/{recording_id}/export.json")
def export_json(recording_id: int, user: User = Depends(require_doctor), db: Session = Depends(get_db)):
    r = db.get(EEGRecording, recording_id)
    if not r: raise HTTPException(404, "Recording not found")
    p = db.get(Patient, r.patient_id)
    if not p or p.owner_doctor_id != user.id: raise HTTPException(403, "Not authorized")
    return {"recording": recording_dict(r), "analysis": run_model(), "seizure_events": [], "note": "Events unavailable from current model."}

@app.get("/api/analysis/{recording_id}/export.csv")
def export_csv(recording_id: int, user: User = Depends(require_doctor), db: Session = Depends(get_db)):
    r = db.get(EEGRecording, recording_id)
    if not r: raise HTTPException(404, "Recording not found")
    p = db.get(Patient, r.patient_id)
    if not p or p.owner_doctor_id != user.id: raise HTTPException(403, "Not authorized")
    channels, matrix = load_matrix(r)
    import io
    out = io.StringIO(); writer = csv.writer(out); writer.writerow(channels)
    for row in matrix[:100000]: writer.writerow(["" if not __import__("math").isfinite(float(v)) else float(v) for v in row])
    out.seek(0)
    return StreamingResponse(iter([out.getvalue()]), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="recording_{recording_id}_channels.csv"'})
