import React, { useState } from "react";
import "./style.css";

const API_URL = "http://127.0.0.1:8000/api";

function readableApiError(detail, fallback = "Something went wrong. Please try again.") {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => {
      if (typeof item === "string") return item;
      const location = Array.isArray(item?.loc) ? item.loc.filter((part) => part !== "body").join(" → ") : "";
      const message = item?.msg || item?.message || "Invalid input";
      return location ? `${location}: ${message}` : message;
    }).join("; ");
  }
  if (detail && typeof detail === "object") {
    if (typeof detail.detail === "string") return detail.detail;
    if (typeof detail.message === "string") return detail.message;
    try { return JSON.stringify(detail); } catch { return fallback; }
  }
  return fallback;
}
export default function App() {
  const [mode, setMode] = useState("");
  const [page, setPage] = useState("login");
  const [pageHistory, setPageHistory] = useState([]);

  // Track internal navigation so Back returns to the actual previous page.
  function navigateTo(nextPage) {
    if (nextPage !== page) {
      setPageHistory((history) => [...history, page]);
      setPage(nextPage);
    }
    if (nextPage === "upload") {
      loadPatients();
    }
  }

  function goBack() {
    if (pageHistory.length > 0) {
      const previous = pageHistory[pageHistory.length - 1];
      setPageHistory((history) => history.slice(0, -1));
      setPage(previous);
    } else {
      setPage(role === "parent" ? "parent" : "doctor");
    }
  }
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [registerName, setRegisterName] = useState("");
  const [isRegistering, setIsRegistering] = useState(false);
  const [token, setToken] = useState(
    localStorage.getItem("neuroai_token") || ""
  );
  const [role, setRole] = useState(
    localStorage.getItem("neuroai_role") || ""
  );
  const [user, setUser] = useState(null);

  const [patients, setPatients] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [eegData, setEegData] = useState(null);
  const [babyProfiles, setBabyProfiles] = useState([]);
  const [babyPhotoPreview, setBabyPhotoPreview] = useState("");
  const [babyFormMessage, setBabyFormMessage] = useState("");
  const [parentReports, setParentReports] = useState([]);
  const [reportFile, setReportFile] = useState(null);
  const [reportTitle, setReportTitle] = useState("");
  const [reportNotes, setReportNotes] = useState("");
  const [reportMessage, setReportMessage] = useState("");

  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  async function registerAccount(event) {
    event.preventDefault();
    if (mode !== "parent") {
      setMessage("Public account creation is currently available for Parent Mode only.");
      return;
    }
    setLoading(true);
    setMessage("");
    try {
      const response = await fetch(`${API_URL}/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: registerName.trim(), email: email.trim(), password }),
      });
      const responseText = await response.text();
      let data = {};
      try { data = responseText ? JSON.parse(responseText) : {}; }
      catch { data = { detail: responseText || "Unexpected server response." }; }
      if (!response.ok) throw new Error(readableApiError(data.detail ?? data, "Account creation failed."));
      setIsRegistering(false);
      setPassword("");
      setMessage("Parent account created successfully. Please sign in with your new credentials.");
    } catch (error) {
      setMessage(readableApiError(error?.message, "Unable to create account. Check the backend response."));
    } finally {
      setLoading(false);
    }
  }

  async function login(event) {
    event.preventDefault();

    if (!mode) {
      setMessage("Please select Doctor Mode or Parent Mode.");
      return;
    }

    setLoading(true);
    setMessage("");

    try {
      // The NeuroAI FastAPI backend expects JSON with email and password.
      const response = await fetch(`${API_URL}/auth/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email: email.trim(),
          password,
        }),
      });

      // Read the response safely, including non-JSON error responses.
      const responseText = await response.text();
      let data = {};
      try {
        data = responseText ? JSON.parse(responseText) : {};
      } catch {
        data = { detail: responseText || "Unexpected server response." };
      }

      if (!response.ok) {
        throw new Error(readableApiError(data.detail ?? data, "Login failed."));
      }

      const accessToken = data.access_token;
      const accountRole = String(
        data.role || data.user?.role || ""
      ).toLowerCase();

      if (!accessToken) {
        throw new Error(
          "The server did not return an access token."
        );
      }

      /*
       * The selected mode is only a preference.
       * The authenticated server-side role must determine
       * which dashboard the user can access.
       */
      const normalizedRole =
        accountRole === "doctor"
          ? "doctor"
          : accountRole === "parent" ||
              accountRole === "patient"
            ? "parent"
            : "";

      if (!normalizedRole) {
        throw new Error(
          "The server did not return a supported account role. " +
            "Check the backend login response."
        );
      }

      if (normalizedRole !== mode) {
        throw new Error(
          `This account is registered for ${normalizedRole} mode. ` +
            "Please select the matching mode."
        );
      }

      localStorage.setItem("neuroai_token", accessToken);
      localStorage.setItem("neuroai_role", normalizedRole);

      setToken(accessToken);
      setRole(normalizedRole);
      setUser(data.user || { email });
      setPage(
        normalizedRole === "doctor"
          ? "doctor"
          : "parent"
      );
      setMessage("");
    } catch (error) {
      setMessage(
        readableApiError(error?.message, "Unable to connect to the NeuroAI server.")
      );
    } finally {
      setLoading(false);
    }
  }

  function logout() {
    localStorage.removeItem("neuroai_token");
    localStorage.removeItem("neuroai_role");

    setToken("");
    setRole("");
    setUser(null);
    setEmail("");
    setPassword("");
    setMode("");
    setPatients([]);
    setEegData(null);
    setPageHistory([]);
    setPage("login");
  }

  async function loadPatients() {
    setLoading(true);
    setMessage("");

    try {
      const response = await fetch(`${API_URL}/patients`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(readableApiError(data.detail ?? data, "Unable to load patients."));
      }

      setPatients(Array.isArray(data) ? data : data.items || []);
    } catch (error) {
      setMessage(readableApiError(error?.message, "Patient data could not be loaded."));
    } finally {
      setLoading(false);
    }
  }

  async function uploadEEG(event) {
    event.preventDefault();

    if (!selectedPatientId) {
      setMessage("Please select a patient before uploading the EEG.");
      return;
    }

    if (!selectedFile) {
      setMessage("Please select an EEG file first.");
      return;
    }

    const formData = new FormData();
    // IMPORTANT: backend expects Patient.id (internal numeric database ID),
    // not Patient.patient_id (the human-readable identifier).
    formData.append("patient_id", String(selectedPatientId));
    formData.append("file", selectedFile);

    setLoading(true);
    setMessage("");
    setEegData(null);

    try {
      const response = await fetch(`${API_URL}/eeg/upload`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          readableApiError(data.detail ?? data, "EEG upload failed.")
        );
      }

      setEegData(data);
      const patient = patients.find(
        (item) => String(item.id) === String(selectedPatientId)
      );
      setMessage(
        `EEG uploaded successfully for ${patient?.name || `patient ${selectedPatientId}`}.`
      );
    } catch (error) {
      setMessage(
        readableApiError(
          error?.message,
          "Unable to upload the EEG recording."
        )
      );
    } finally {
      setLoading(false);
    }
  }

  async function loadBabyProfiles() {
    if (!token || role !== "parent") return;
    try {
      const response = await fetch(`${API_URL}/parent/babies`, { headers: { Authorization: `Bearer ${token}` } });
      const data = await response.json();
      if (!response.ok) throw new Error(readableApiError(data.detail, "Unable to load baby profiles."));
      const items = Array.isArray(data) ? data : [];
      const withPhotos = await Promise.all(items.map(async (baby) => {
        if (!baby.photo_url) return baby;
        try {
          const photoResponse = await fetch(`http://127.0.0.1:8000${baby.photo_url}`, { headers: { Authorization: `Bearer ${token}` } });
          if (!photoResponse.ok) return baby;
          const blob = await photoResponse.blob();
          return { ...baby, photo_data_url: URL.createObjectURL(blob) };
        } catch { return baby; }
      }));
      setBabyProfiles(withPhotos);
    } catch (error) { setBabyFormMessage(error.message || "Unable to load baby profiles."); }
  }

  async function createBabyProfile(event) {
    event.preventDefault();
    setLoading(true); setBabyFormMessage("");
    try {
      const form = event.currentTarget;
      const formData = new FormData(form);
      const photo = formData.get("photo");
      if (!photo || !photo.name) formData.delete("photo");
      const response = await fetch(`${API_URL}/parent/babies`, { method: "POST", headers: { Authorization: `Bearer ${token}` }, body: formData });
      const text = await response.text(); let data = {};
      try { data = text ? JSON.parse(text) : {}; } catch { data = { detail: text }; }
      if (!response.ok) throw new Error(readableApiError(data.detail ?? data, "Unable to save baby profile."));
      setBabyFormMessage("Baby profile saved successfully.");
      form.reset(); setBabyPhotoPreview("");
      await loadBabyProfiles();
      setPageHistory((history) => history.length ? history.slice(0, -1) : history);
      setPage("parent");
    } catch (error) { setBabyFormMessage(readableApiError(error?.message, "Unable to save baby profile.")); }
    finally { setLoading(false); }
  }

  async function loadParentReports() {
    if (!token || role !== "parent") return;
    try {
      const response = await fetch(`${API_URL}/parent/reports`, { headers: { Authorization: `Bearer ${token}` } });
      const text = await response.text();
      let data = {}; try { data = text ? JSON.parse(text) : []; } catch { data = { detail: text }; }
      if (!response.ok) throw new Error(readableApiError(data.detail ?? data, "Unable to load uploaded reports."));
      setParentReports(Array.isArray(data) ? data : []);
    } catch (error) { setReportMessage(readableApiError(error?.message, "Unable to load uploaded reports.")); }
  }

  async function uploadParentReport(event) {
    event.preventDefault();
    if (!reportFile) { setReportMessage("Please choose a PDF or image file first."); return; }
    const formData = new FormData();
    formData.append("file", reportFile);
    formData.append("title", reportTitle.trim() || reportFile.name);
    formData.append("notes", reportNotes.trim());
    setLoading(true); setReportMessage("");
    try {
      const response = await fetch(`${API_URL}/parent/reports`, { method: "POST", headers: { Authorization: `Bearer ${token}` }, body: formData });
      const text = await response.text(); let data = {};
      try { data = text ? JSON.parse(text) : {}; } catch { data = { detail: text }; }
      if (!response.ok) throw new Error(readableApiError(data.detail ?? data, "Unable to upload report."));
      setReportMessage("Report document uploaded successfully. It is stored as a parent-provided document and has not been clinically verified.");
      setReportFile(null); setReportTitle(""); setReportNotes("");
      const input = document.getElementById("parent-report-file"); if (input) input.value = "";
      await loadParentReports();
    } catch (error) { setReportMessage(readableApiError(error?.message, "Unable to upload report.")); }
    finally { setLoading(false); }
  }

  async function downloadParentReport(report) {
    try {
      const response = await fetch(`${API_URL}/parent/reports/${report.id}/download`, { headers: { Authorization: `Bearer ${token}` } });
      if (!response.ok) { const text = await response.text(); let data = {}; try { data = JSON.parse(text); } catch { data = { detail: text }; } throw new Error(readableApiError(data.detail ?? data, "Unable to download report.")); }
      const blob = await response.blob(); const url = URL.createObjectURL(blob);
      const link = document.createElement("a"); link.href = url; link.download = report.original_filename || `report-${report.id}`; document.body.appendChild(link); link.click(); link.remove(); URL.revokeObjectURL(url);
    } catch (error) { setReportMessage(readableApiError(error?.message, "Unable to download report.")); }
  }

  if (!token || !role || page === "login") {
    return (
      <main className="login-page">
        <header className="brand">
          <span className="brand-icon">N</span>
          <span>
            NeuroAI<span className="brand-dot">.</span>
          </span>
        </header>

        <section className="login-layout">
          <div className="welcome-panel">
            <span className="eyebrow">
              NEONATAL EEG INTELLIGENCE
            </span>

            <h1>
              Better insights.
              <br />
              <span>Gentler care.</span>
            </h1>

            <p>
              An AI-assisted platform for neonatal EEG
              review, clinical reporting, and care-team
              collaboration.
            </p>

            <div className="welcome-note">
              <span className="note-icon">♡</span>
              <div>
                <strong>Designed around every baby</strong>
                <p>
                  Clear information for clinicians and
                  families.
                </p>
              </div>
            </div>

            <div className="decorative-circle circle-one" />
            <div className="decorative-circle circle-two" />
          </div>

          <div className="login-card">
            {mode && (
              <button
                type="button"
                className="back-button"
                onClick={() => {
                  setMode("");
                  setIsRegistering(false);
                  setMessage("");
                }}
              >
                ← Back
              </button>
            )}

            <span className="eyebrow">WELCOME TO NEUROAI</span>

            <h2>Let's get started</h2>

            <p className="muted">
              Choose your portal to continue.
            </p>

            <div className="mode-grid">
              <button
                type="button"
                className={`mode-card doctor-card ${
                  mode === "doctor" ? "selected" : ""
                }`}
                onClick={() => {
                  setMode("doctor");
                  setMessage("");
                }}
              >
                <span className="mode-icon">✚</span>
                <strong>Doctor Mode</strong>
                <small>
                  Clinical dashboard and EEG review
                </small>
                <span className="mode-arrow">→</span>
              </button>

              <button
                type="button"
                className={`mode-card parent-card ${
                  mode === "parent" ? "selected" : ""
                }`}
                onClick={() => {
                  setMode("parent");
                  setMessage("");
                }}
              >
                <span className="mode-icon">♡</span>
                <strong>Parent Mode</strong>
                <small>
                  Baby's reports and shared instructions
                </small>
                <span className="mode-arrow">→</span>
              </button>
            </div>

            <form onSubmit={isRegistering ? registerAccount : login}>
              {isRegistering && (
                <>
                  <label htmlFor="register-name">Full name</label>
                  <input id="register-name" type="text" placeholder="Your full name" value={registerName} onChange={(event) => setRegisterName(event.target.value)} required minLength={2} />
                </>
              )}
              <label htmlFor="email">Email address</label>

              <input
                id="email"
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(event) =>
                  setEmail(event.target.value)
                }
                required
              />

              <label htmlFor="password">Password</label>

              <input
                id="password"
                type="password"
                placeholder="Enter your password"
                value={password}
                onChange={(event) =>
                  setPassword(event.target.value)
                }
                required
              />

              {message && (
                <div className="message" role="alert">
                  {message}
                </div>
              )}

              <button
                className="primary-button"
                type="submit"
                disabled={loading || !mode}
              >
                {loading ? (isRegistering ? "Creating account..." : "Signing in...") : isRegistering ? "Create Parent Account" : mode ? `Continue as ${mode === "doctor" ? "Doctor" : "Parent"}` : "Select a mode to continue"}
              </button>
              {mode === "parent" && (
                <button type="button" className="text-button" onClick={() => { setIsRegistering((value) => !value); setMessage(""); }}>
                  {isRegistering ? "Already have an account? Sign in" : "New here? Create a parent account"}
                </button>
              )}
            </form>

            <p className="security-note">
              🔒 Your account determines which information
              you can access.
            </p>

            <p className="disclaimer">
              NeuroAI supports clinical review. It does not
              independently diagnose medical conditions.
            </p>
          </div>
        </section>
      </main>
    );
  }

  if (role === "parent") {
    return (
      <div className="app-shell">
        <aside className="sidebar parent-sidebar">
          <div className="brand sidebar-brand">
            <span className="brand-icon">N</span>
            <span>
              NeuroAI<span className="brand-dot">.</span>
            </span>
          </div>

          <div className="sidebar-label">PARENT PORTAL</div>

          <button
            className="nav-item active"
            onClick={() => navigateTo("parent")}
          >
            ♡ &nbsp; Overview
          </button>

          <button
            className={`nav-item ${page === "parent-add-baby" ? "active" : ""}`}
            onClick={() => { navigateTo("parent-add-baby"); setBabyFormMessage(""); loadBabyProfiles(); }}
          >
            ♧ &nbsp; Baby Profile
          </button>

          <button
            className="nav-item"
            onClick={() => { navigateTo("parent-reports"); loadBabyProfiles(); loadParentReports(); setReportMessage(""); }}
          >
            ▤ &nbsp; EEG Reports
          </button>

          <button
            className="nav-item"
            onClick={() => navigateTo("parent-instructions")}
          >
            ♧ &nbsp; Care Instructions
          </button>

          <div className="sidebar-bottom">
            <button
              className="nav-item"
              onClick={logout}
            >
              ↪ &nbsp; Log out
            </button>
          </div>
        </aside>

        <main className="main-content">
          <header className="topbar">
            <div>
              <span className="eyebrow">
                PARENT PORTAL
              </span>
              <h1>Welcome to your baby's care space ♡</h1>
            </div>

            <div className="avatar parent-avatar">P</div>
          </header>

          <div className="page-back-row">
            <button type="button" className="back-button" onClick={goBack}>
              ← Back
            </button>
          </div>

          {page === "parent" && (
            <>
              <section className="hero-banner parent-banner">
                <div>
                  <span className="eyebrow">
                    YOUR FAMILY'S CARE SPACE
                  </span>

                  <h2>
                    Every little step
                    <br />
                    matters.
                  </h2>

                  <p>
                    View reports and information your
                    healthcare team has shared with you.
                  </p>
                </div>

                <div className="hero-heart">♡</div>
              </section>

              <section className="stats-grid">
                <div className="stat-card">
                  <span className="stat-icon pink-icon">
                    ♡
                  </span>
                  <span>Baby profile</span>
                  <strong>{babyProfiles.length || "No profile"}</strong>
                  <small>Baby profile(s) saved to your parent account.</small>
                  <button type="button" className="text-action" onClick={() => { navigateTo("parent-add-baby"); loadBabyProfiles(); }}>+ Add baby's details</button>
                </div>

                <div className="stat-card">
                  <span className="stat-icon blue-icon">
                    ▤
                  </span>
                  <span>Available reports</span>
                  <strong>—</strong>
                  <small>
                    Reports appear after secure account
                    linking.
                  </small>
                </div>
              </section>

              <section className="content-card">
                <div className="section-heading"><div><h2>Your baby's profile</h2><p>Add and maintain the details you want to keep in one place.</p></div><button className="primary-button compact" type="button" onClick={() => { navigateTo("parent-add-baby"); setBabyFormMessage(""); }}>+ Add baby</button></div>
                {babyProfiles.length === 0 ? <div className="empty-state"><span>♡</span><strong>No baby profiles yet</strong><p>Add your baby's details and a photo to create a family profile.</p><button className="primary-button compact" type="button" onClick={() => navigateTo("parent-add-baby")}>Add baby's details</button></div> : <div className="baby-profile-grid">{babyProfiles.map((baby) => <article className="baby-profile-card" key={baby.id}>{baby.photo_url ? <img className="baby-profile-photo" src={baby.photo_data_url || ""} alt={`Photo of ${baby.full_name}`} /> : <div className="baby-profile-photo baby-photo-placeholder">♡</div>}<div className="baby-profile-body"><h3>{baby.full_name}</h3><p>{baby.date_of_birth || "Date of birth not provided"}{baby.sex ? ` · ${baby.sex}` : ""}</p><div className="baby-detail-chips">{baby.blood_group && <span>Blood group: {baby.blood_group}</span>}{baby.birth_weight_kg != null && <span>Birth weight: {baby.birth_weight_kg} kg</span>}{baby.gestational_age_weeks != null && <span>Gestation: {baby.gestational_age_weeks}w {baby.gestational_age_days || 0}d</span>}</div><small>{baby.information_notice}</small></div></article>)}</div>}
                <div className="notice-box"><strong>Clinical record note</strong><p>These details are entered by the parent and are not clinically verified. They do not automatically link the profile to a hospital patient record or grant access to clinical reports.</p></div>
              </section>
            </>
          )}

          {page === "parent-add-baby" && (
            <section className="content-card baby-form-card">
              <span className="eyebrow">FAMILY PROFILE</span><h2>Add your baby's details</h2>
              <p>Fill in what you know. Only the baby's name is required; you can add or update details later by creating another profile. Please avoid adding unnecessary sensitive information.</p>
              {babyFormMessage && <div className={babyFormMessage.toLowerCase().includes("success") ? "notice-box" : "message"} role="status">{babyFormMessage}</div>}
              <form className="baby-form" onSubmit={createBabyProfile} encType="multipart/form-data">
                <div className="baby-photo-upload"><div className="baby-photo-preview">{babyPhotoPreview ? <img src={babyPhotoPreview} alt="Baby preview" /> : <span>♡</span>}</div><div><label htmlFor="baby-photo">Baby photo (JPG, PNG or WEBP; max 5 MB)</label><input id="baby-photo" name="photo" type="file" accept="image/jpeg,image/png,image/webp" onChange={(e) => { const file=e.target.files?.[0]; if(file){ if(file.size > 5*1024*1024){setBabyFormMessage("Photo must be 5 MB or smaller."); e.target.value=""; return;} setBabyPhotoPreview(URL.createObjectURL(file)); } else setBabyPhotoPreview(""); }} /></div></div>
                <h3>Basic details</h3><div className="baby-fields-grid">
                  <label>Baby's full name *<input name="full_name" required maxLength="120" placeholder="Full name" /></label>
                  <label>Date of birth<input name="date_of_birth" type="date" /></label>
                  <label>Time of birth<input name="birth_time" type="time" /></label>
                  <label>Sex/gender<select name="sex" defaultValue=""><option value="">Select</option><option>Female</option><option>Male</option><option>Intersex</option><option>Prefer not to say</option></select></label>
                  <label>Blood group<select name="blood_group" defaultValue=""><option value="">Unknown / not tested</option>{["A+","A-","B+","B-","AB+","AB-","O+","O-"].map(v=><option key={v}>{v}</option>)}</select></label>
                  <label>Birth type<select name="birth_type" defaultValue=""><option value="">Select</option><option>Vaginal delivery</option><option>Caesarean section</option><option>Assisted delivery</option><option>Other / unknown</option></select></label>
                </div>
                <h3>Birth measurements</h3><div className="baby-fields-grid">
                  <label>Gestational age — weeks<input name="gestational_age_weeks" type="number" min="0" max="45" placeholder="e.g. 36" /></label>
                  <label>Gestational age — days<input name="gestational_age_days" type="number" min="0" max="6" placeholder="0–6" /></label>
                  <label>Birth weight (kg)<input name="birth_weight_kg" type="number" min="0" max="10" step="0.001" placeholder="e.g. 2.85" /></label>
                  <label>Birth length (cm)<input name="birth_length_cm" type="number" min="0" max="100" step="0.1" /></label>
                  <label>Head circumference (cm)<input name="head_circumference_cm" type="number" min="0" max="60" step="0.1" /></label>
                  <label>Birth status<select name="birth_status" defaultValue=""><option value="">Select</option><option>Stable</option><option>NICU admission</option><option>Special care nursery</option><option>Other / not specified</option></select></label>
                </div>
                <h3>Hospital and care team</h3><div className="baby-fields-grid">
                  <label>Hospital name<input name="hospital_name" placeholder="Hospital / clinic" /></label>
                  <label>Hospital unit<input name="hospital_unit" placeholder="e.g. NICU" /></label>
                  <label>Admission date<input name="admission_date" type="date" /></label>
                  <label>Discharge date<input name="discharge_date" type="date" /></label>
                  <label>Pediatrician / clinician name<input name="pediatrician_name" placeholder="Clinician name" /></label>
                  <label>Clinician phone<input name="pediatrician_phone" type="tel" placeholder="Phone number" /></label>
                </div>
                <h3>Parent and emergency contact</h3><div className="baby-fields-grid">
                  <label>Your relationship to baby<select name="parent_relationship" defaultValue=""><option value="">Select</option><option>Mother</option><option>Father</option><option>Guardian</option><option>Other caregiver</option></select></label>
                  <label>Emergency contact name<input name="emergency_contact_name" placeholder="Contact name" /></label>
                  <label>Emergency contact phone<input name="emergency_contact_phone" type="tel" placeholder="Phone number" /></label>
                </div>
                <h3>Health information</h3><div className="baby-fields-grid baby-fields-single">
                  <label>Known allergies<textarea name="allergies" rows="2" placeholder="Write known allergies, or 'None known'" /></label>
                  <label>Current medications<textarea name="current_medications" rows="2" placeholder="Medicine, dose, and schedule if known" /></label>
                  <label>Medical history / diagnoses<textarea name="medical_history" rows="3" placeholder="Relevant history as provided by your healthcare team" /></label>
                  <label>Pregnancy, birth or additional notes<textarea name="pregnancy_birth_notes" rows="3" placeholder="Optional additional information" /></label>
                </div>
                <div className="notice-box"><strong>Privacy and accuracy</strong><p>This profile is parent-entered, not a medical diagnosis or verified hospital record. Only upload a photo you are authorized to share. For urgent concerns, contact your healthcare team.</p></div>
                <div className="baby-form-actions"><button type="button" className="secondary-button" onClick={goBack}>Cancel</button><button className="primary-button" type="submit" disabled={loading}>{loading ? "Saving profile…" : "Save baby's profile"}</button></div>
              </form>
            </section>
          )}

          {page === "parent-reports" && (
            <section className="content-card parent-reports-page">
              <h2>EEG Reports & Documents</h2>
              <p>Upload a copy of an EEG report provided by your healthcare team, and keep it with your family records.</p>
              <div className="notice-box report-upload-notice"><strong>Important</strong><p>Uploaded files are stored as parent-provided documents. Uploading does not analyze EEG signals, verify the report, or share it with your clinician. Only upload documents you are authorized to store.</p></div>
              <form className="parent-report-upload" onSubmit={uploadParentReport}>
                <h3>Upload a report</h3>
                <label htmlFor="parent-report-title">Report title (optional)</label>
                <input id="parent-report-title" type="text" maxLength="180" placeholder="e.g. EEG report — 11 October 2026" value={reportTitle} onChange={(event) => setReportTitle(event.target.value)} />
                <label htmlFor="parent-report-file">Choose report file</label>
                <input id="parent-report-file" type="file" accept=".pdf,.jpg,.jpeg,.png,.webp,application/pdf,image/jpeg,image/png,image/webp" onChange={(event) => setReportFile(event.target.files?.[0] || null)} required />
                <small>Accepted formats: PDF, JPG, PNG, WEBP. Maximum file size: 10 MB.</small>
                <label htmlFor="parent-report-notes">Notes (optional)</label>
                <textarea id="parent-report-notes" rows="3" placeholder="Optional notes, such as the visit date. Avoid adding unnecessary sensitive details." value={reportNotes} onChange={(event) => setReportNotes(event.target.value)} />
                {reportMessage && <div className="message" role="status">{reportMessage}</div>}
                <button className="primary-button compact" type="submit" disabled={loading}>{loading ? "Uploading report…" : "Upload report"}</button>
              </form>
              <div className="parent-report-list-heading"><h3>Your uploaded documents</h3><button type="button" className="secondary-button" onClick={loadParentReports}>Refresh</button></div>
              {parentReports.length === 0 ? (
                <div className="empty-state"><span>▤</span><strong>No uploaded documents yet</strong><p>Files you upload will appear here. Reports shared directly by the care team require a separate secure linking feature.</p></div>
              ) : (
                <div className="parent-report-list">{parentReports.map((report) => (
                  <article className="parent-report-row" key={report.id}>
                    <span className="report-file-icon">▤</span><div className="parent-report-details"><strong>{report.title}</strong><span>{report.original_filename} · {(report.file_size / (1024 * 1024)).toFixed(2)} MB</span><small>{report.created_at ? new Date(report.created_at).toLocaleString() : "Recently uploaded"} · Parent-provided, not clinically verified</small>{report.notes && <p>{report.notes}</p>}</div>
                    <button type="button" className="secondary-button" onClick={() => downloadParentReport(report)}>Download</button>
                  </article>
                ))}</div>
              )}
            </section>
          )}

          {page === "parent-instructions" && (
            <section className="content-card">
              <h2>Care Instructions</h2>
              <p>
                Instructions specifically shared with the
                parent account will appear here once the
                relevant backend feature is connected.
              </p>
              <div className="notice-box">
                For medical advice, contact your baby's
                treating clinician.
              </div>
            </section>
          )}
        </main>
      </div>
    );
  }

  if (role === "doctor") {
    return (
      <div className="app-shell">
        <aside className="sidebar">
          <div className="brand sidebar-brand">
            <span className="brand-icon">N</span>
            <span>
              NeuroAI<span className="brand-dot">.</span>
            </span>
          </div>

          <div className="sidebar-label">
            CLINICAL WORKSPACE
          </div>

          <button
            className={`nav-item ${
              page === "doctor" ? "active" : ""
            }`}
            onClick={() => navigateTo("doctor")}
          >
            ▦ &nbsp; Dashboard
          </button>

          <button
            className="nav-item"
            onClick={() => {
              navigateTo("patients");
              loadPatients();
            }}
          >
            ♙ &nbsp; Patients
          </button>

          <button
            className="nav-item"
            onClick={() => navigateTo("upload")}
          >
            〰 &nbsp; EEG Upload
          </button>

          <button
            className="nav-item"
            onClick={() => navigateTo("reports")}
          >
            ▤ &nbsp; Reports
          </button>

          <div className="sidebar-bottom">
            <button
              className="nav-item"
              onClick={logout}
            >
              ↪ &nbsp; Log out
            </button>
          </div>
        </aside>

        <main className="main-content">
          <header className="topbar">
            <div>
              <span className="eyebrow">
                CLINICAL WORKSPACE
              </span>
              <h1>Good evening, Doctor 👋</h1>
              <p className="muted">
                Welcome back to your neonatal EEG dashboard.
              </p>
            </div>

            <div className="avatar">D</div>
          </header>

          <div className="page-back-row">
            <button type="button" className="back-button" onClick={goBack}>
              ← Back
            </button>
          </div>

          {message && (
            <div className="message">{message}</div>
          )}

          {page === "doctor" && (
            <>
              <section className="hero-banner doctor-banner">
                <div>
                  <span className="eyebrow">
                    NEUROAI OVERVIEW
                  </span>

                  <h2>
                    Clinical insights,
                    <br />
                    thoughtfully organized.
                  </h2>

                  <p>
                    Review recordings and manage patient
                    information from one workspace.
                  </p>

                  <button
                    className="primary-button compact"
                    onClick={() => navigateTo("upload")}
                  >
                    Upload EEG recording →
                  </button>
                </div>

                <div className="hero-symbol">〰</div>
              </section>

              <section className="stats-grid">
                <div className="stat-card">
                  <span className="stat-icon blue-icon">
                    ♙
                  </span>
                  <span>Total patients</span>
                  <strong>—</strong>
                  <small>Load patient records to view totals.</small>
                </div>

                <div className="stat-card">
                  <span className="stat-icon pink-icon">
                    〰
                  </span>
                  <span>EEG recordings</span>
                  <strong>—</strong>
                  <small>Recording totals are not loaded.</small>
                </div>

                <div className="stat-card">
                  <span className="stat-icon blue-icon">
                    ▤
                  </span>
                  <span>Completed analyses</span>
                  <strong>—</strong>
                  <small>
                    Requires actual analysis records.
                  </small>
                </div>

                <div className="stat-card">
                  <span className="stat-icon pink-icon">
                    ♡
                  </span>
                  <span>Seizure events</span>
                  <strong>—</strong>
                  <small>
                    Not available from the current model.
                  </small>
                </div>
              </section>

              <section className="content-card">
                <div className="section-heading">
                  <div>
                    <h2>Quick actions</h2>
                    <p>
                      Continue with a clinical workflow.
                    </p>
                  </div>
                </div>

                <div className="quick-actions">
                  <button
                    className="action-card"
                    onClick={() => {
                      navigateTo("patients");
                      loadPatients();
                    }}
                  >
                    <span className="action-icon">♙</span>
                    <strong>View patients</strong>
                    <small>Open patient records</small>
                  </button>

                  <button
                    className="action-card"
                    onClick={() => navigateTo("upload")}
                  >
                    <span className="action-icon">〰</span>
                    <strong>Upload EEG</strong>
                    <small>Choose a CSV recording</small>
                  </button>

                  <button
                    className="action-card"
                    onClick={() => navigateTo("reports")}
                  >
                    <span className="action-icon">▤</span>
                    <strong>Reports</strong>
                    <small>Review available reports</small>
                  </button>
                </div>
              </section>

              <section className="content-card">
                <h2>Scientific integrity</h2>
                <p>
                  Seizure predictions, confidence scores,
                  and localization are not displayed unless
                  the connected model actually provides
                  them.
                </p>
                <div className="notice-box">
                  AI-assisted clinical support only. All
                  results require appropriate clinical
                  review.
                </div>
              </section>
            </>
          )}

          {page === "patients" && (
            <section className="content-card">
              <h2>Patient records</h2>
              <p>
                Records returned by the backend are shown
                below.
              </p>

              {patients.length === 0 ? (
                <div className="empty-state">
                  <span>♙</span>
                  <strong>No patient records loaded</strong>
                  <p>
                    Check the backend patient endpoint and
                    your account permissions.
                  </p>
                </div>
              ) : (
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Patient</th>
                        <th>Patient ID</th>
                        <th>Details</th>
                      </tr>
                    </thead>

                    <tbody>
                      {patients.map((patient, index) => (
                        <tr key={patient.id || index}>
                          <td>
                            {patient.name ||
                              patient.full_name ||
                              "Unnamed patient"}
                          </td>
                          <td>
                            {patient.patient_id ||
                              patient.id ||
                              "—"}
                          </td>
                          <td>
                            {patient.age
                              ? `${patient.age} days`
                              : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          )}

          {page === "upload" && (
            <section className="content-card">
              <span className="eyebrow">EEG WORKSPACE</span>
              <h2>Upload EEG recording</h2>
              <p>
                Choose the patient first, then select an EDF, CSV, or TSV recording.
              </p>

              <form onSubmit={uploadEEG}>
                <label htmlFor="patient-select">Select patient</label>
                <select
                  id="patient-select"
                  value={selectedPatientId}
                  onChange={(event) => setSelectedPatientId(event.target.value)}
                  required
                >
                  <option value="">-- Choose a patient --</option>
                  {patients.map((patient) => (
                    <option key={patient.id} value={patient.id}>
                      {patient.name || patient.full_name || "Unnamed patient"} — ID: {patient.patient_id || patient.id}
                    </option>
                  ))}
                </select>

                {patients.length === 0 && (
                  <div className="notice-box">
                    No patients loaded. Create a patient first or click the button below.
                    <button
                      type="button"
                      className="primary-button compact"
                      disabled={loading}
                      onClick={loadPatients}
                    >
                      Refresh patients
                    </button>
                  </div>
                )}

                <label htmlFor="eeg-file">
                  EEG recording (EDF, CSV, or TSV)
                </label>

                <input
                  id="eeg-file"
                  type="file"
                  accept=".edf,.csv,.tsv"
                  required
                  onChange={(event) =>
                    setSelectedFile(event.target.files?.[0] || null)
                  }
                />

                {selectedFile && (
                  <p className="file-info">
                    Selected: {selectedFile.name}
                  </p>
                )}

                <button
                  className="primary-button compact"
                  disabled={loading}
                  type="submit"
                >
                  {loading
                    ? "Uploading..."
                    : "Upload recording"}
                </button>
              </form>

              {eegData && (
                <div className="notice-box">
                  <strong>Server response</strong>
                  <pre className="response-json">
                    {JSON.stringify(eegData, null, 2)}
                  </pre>
                </div>
              )}

              <p className="disclaimer">
                Uploading a file does not by itself produce
                a validated seizure diagnosis.
              </p>
            </section>
          )}

          {page === "reports" && (
            <section className="content-card">
              <h2>Clinical reports</h2>
              <p>
                Report generation and retrieval depend on
                the backend report endpoints.
              </p>

              <div className="empty-state">
                <span>▤</span>
                <strong>No reports loaded</strong>
                <p>
                  Generate or retrieve reports through the
                  configured backend.
                </p>
              </div>
            </section>
          )}
        </main>
      </div>
    );
  }

  return (
    <main className="login-page">
      <section className="login-card">
        <h1>Account role unavailable</h1>
        <p>
          Your account role could not be verified.
          Please sign in again.
        </p>
        <button className="primary-button" onClick={logout}>
          Return to login
        </button>
      </section>
    </main>
  );
}