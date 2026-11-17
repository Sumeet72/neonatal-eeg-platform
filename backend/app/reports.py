from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm

DISCLAIMER = ("This report is generated using an AI-assisted EEG analysis system and is intended to support "
"clinical/research review. It must not be used as a standalone diagnosis. Final interpretation must be performed "
"by a qualified healthcare professional.")

def make_report(path: str, patient: dict, recording: dict, analysis_status: str, doctor_name: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(path, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=18*mm, bottomMargin=18*mm)
    styles = getSampleStyleSheet()
    story = [Paragraph("NEONATAL EEG ANALYSIS REPORT", styles["Title"]), Spacer(1, 8*mm),
             Paragraph("Patient Information", styles["Heading2"])]
    patient_rows = [["Patient ID", patient.get("patient_id", "—")], ["Name", patient.get("name", "—")],
                    ["Age", patient.get("age") or "—"], ["Gender", patient.get("gender") or "—"],
                    ["Gestational age", patient.get("gestational_age") or "—"],
                    ["Birth weight (kg)", str(patient.get("birth_weight") or "—")]]
    t = Table(patient_rows, colWidths=[48*mm, 115*mm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-1),colors.HexColor("#eef0ff")),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#dce0eb")),("PADDING",(0,0),(-1,-1),7)]))
    story += [t, Spacer(1, 6*mm), Paragraph("EEG Information", styles["Heading2"])]
    eeg_rows = [["File", recording.get("filename", "—")], ["Channels", str(recording.get("channels", []))],
                ["Samples", str(recording.get("sample_count", "—"))], ["Sampling rate", str(recording.get("sampling_rate") or "Not provided")],
                ["Duration", str(recording.get("duration_seconds") or "Not available")]]
    t2 = Table(eeg_rows, colWidths=[48*mm, 115*mm])
    t2.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-1),colors.HexColor("#eef0ff")),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#dce0eb")),("PADDING",(0,0),(-1,-1),7)]))
    story += [t2, Spacer(1, 6*mm), Paragraph("Analysis Status", styles["Heading2"]),
              Paragraph(f"Model status: {analysis_status}", styles["BodyText"]),
              Paragraph("Seizure events, confidence, seizure burden, and localization are not available because a validated model output is not connected.", styles["BodyText"]),
              Spacer(1, 6*mm), Paragraph("Doctor Section", styles["Heading2"]),
              Paragraph(f"Doctor: {doctor_name}", styles["BodyText"]), Spacer(1, 4*mm),
              Paragraph("Clinical notes and prescriptions must be entered and reviewed by an authorized clinician; none are generated automatically by this report.", styles["BodyText"]),
              Spacer(1, 8*mm), Paragraph("Disclaimer", styles["Heading2"]), Paragraph(DISCLAIMER, styles["BodyText"])]
    doc.build(story)
