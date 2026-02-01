# PPE Compliance Monitoring System (Flask · ONNX · YOLO)

This project is a **modular**, **Windows-friendly** PPE compliance monitoring system:

- **2-step pipeline**: person-gating (YOLOv5n ONNX) → compliance detection (your YOLO ONNX model)
- **Live MJPEG stream** + live compliance status panel
- **Analytics dashboard** with:
  - total / compliant / non-compliant counts
  - compliance rate
  - compliant & non-compliant galleries
  - timestamps, confidence, detected classes
  - click-to-enlarge frame inspection
- **Manual PDF report generation** (non-compliant section first)
- **Manual email sending** of the generated report via SMTP (`.env` credentials)

## 1) Project Structure

```text
ppe_compliance_system_modular/
  app.py
  core/
    config.py
    logging_config.py
    state.py
  camera/
    streamer.py
  detectors/
    person_detector.py
    compliance_detector.py
    yolo_utils.py
  pipeline/
    processor.py
  alerts/
    beeper.py
  reporting/
    pdf_report.py
    email_sender.py
  routes/
    pages.py
    api.py
  templates/
    index.html
    dashboard.html
    live_feed.html
  static/
    css/style.css
    js/live_feed.js
    js/script.js
  reports/
  models/
  .env.example
  requirements.txt
```

## 2) Setup (Windows)

### A) Create venv + install dependencies
```powershell
cd ppe_compliance_system_modular
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### B) Configure environment
```powershell
copy .env.example .env
notepad .env
```

- Use **RTSP_URL** for CCTV/RTSP
- Or **CAMERA_INDEX=0** for webcam
- Place your compliance model at `models\best.onnx` (or set `COMPLIANCE_MODEL_PATH`)

> If you use Gmail: create an **App Password** and put it in `SMTP_PASSWORD`.

## 3) Run the system
```powershell
python app.py
```

Open:
- Home: `http://127.0.0.1:5000/`
- Live feed: `http://127.0.0.1:5000/live`
- Dashboard: `http://127.0.0.1:5000/dashboard`

## 4) Manual report generation & email (from dashboard)

On the dashboard:
- **Download Report** calls `POST /api/report/generate` (returns a PDF)
- **Email Report** calls `POST /api/report/generate` with `{ "email": "someone@domain.com" }`

Email uses `.env`:
- `EMAIL_ENABLED=1`
- `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`

## 5) Key API endpoints

- `GET /video_feed` — MJPEG stream
- `GET /api/cctv/status` — stats + stream state
- `GET /api/cctv/compliance` — latest compliance result
- `GET /api/frames` — dashboard payload (stats + galleries)
- `POST /api/frames/clear` — clear in-memory frames
- `POST /api/report/generate` — generate PDF, optionally email

## 6) Testing Methodologies

### A) Model & pipeline validation
- **Unit tests (rules):** test compliance rule outcomes for different detected-class combinations.
- **Offline inference test:** run a short recorded video (or saved frames) through the pipeline and verify:
  - person gating triggers only when people are present
  - compliance class detections match expected frames
- **Threshold tuning:** vary `COMPLIANCE_CONF_THRES` and monitor false positives / false negatives.

### B) Alert system validation (Windows)
- Force a known non-compliant frame (e.g., show “exposed_hair”) and confirm:
  - warning/critical beep triggers
  - cooldown prevents spamming

### C) Dashboard correctness
- Confirm:
  - counts increment correctly
  - compliance rate matches `compliant / total`
  - gallery shows latest frames and click-to-enlarge opens modal
  - `/api/frames/clear` clears UI after refresh

### D) Report generation
- Trigger report from dashboard:
  - verify **Non-Compliant Frames** section appears first
  - verify timestamps, reasons, classes, and images are present

### E) Email send (SMTP)
- Enable `EMAIL_ENABLED=1`
- Use a valid SMTP account
- Send to your address and confirm attachment opens

## Notes / Common Issues

- **Webcam not opening on Windows**:
  - Try changing `CAMERA_INDEX` (0/1/2...)
  - Close any apps already using the camera (Teams/Zoom/browser)
- **RTSP stutters**:
  - lower `MAX_FPS`
  - use smaller `FRAME_WIDTH/HEIGHT`

---
If you want, I can add a **/api/settings** endpoint and a small UI to tune thresholds in real time.
