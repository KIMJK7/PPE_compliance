from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

@dataclass(frozen=True)
class Settings:
    # Flask
    FLASK_HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT: int = int(os.getenv("FLASK_PORT", "5000"))
    DEBUG: bool = os.getenv("FLASK_DEBUG", "0") == "1"

    # Video source
    # Use ONE:
    #  - RTSP_URL (recommended) OR
    #  - CAMERA_INDEX (0/1/2...) for USB / laptop webcam
    RTSP_URL: str | None = os.getenv("RTSP_URL") or None
    CAMERA_INDEX: int | None = int(os.getenv("CAMERA_INDEX", "-1"))
    CAMERA_INDEX = None if CAMERA_INDEX == -1 else CAMERA_INDEX

    FRAME_WIDTH: int = int(os.getenv("FRAME_WIDTH", "640"))
    FRAME_HEIGHT: int = int(os.getenv("FRAME_HEIGHT", "360"))
    MAX_FPS: int = int(os.getenv("MAX_FPS", "20"))

    # Models
    MODEL_DIR: str = os.getenv("MODEL_DIR", "models")

    # Person gating model (YOLOv5n ONNX)
    PERSON_MODEL_URL: str = os.getenv(
        "PERSON_MODEL_URL",
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx",
    )
    PERSON_MODEL_FILENAME: str = os.getenv("PERSON_MODEL_FILENAME", "yolov5n.onnx")
    PERSON_CONF_THRES: float = float(os.getenv("PERSON_CONF_THRES", "0.50"))

    # Compliance model (YOLO ONNX trained with your classes)
    COMPLIANCE_MODEL_PATH: str = os.getenv("COMPLIANCE_MODEL_PATH", "models/best.onnx")
    # Optional: if COMPLIANCE_MODEL_PATH does not exist, and URL is provided, we download it.
    COMPLIANCE_MODEL_URL: str | None = os.getenv("COMPLIANCE_MODEL_URL") or None
    COMPLIANCE_CONF_THRES: float = float(os.getenv("COMPLIANCE_CONF_THRES", "0.25"))
    COMPLIANCE_IOU_THRES: float = float(os.getenv("COMPLIANCE_IOU_THRES", "0.45"))

    # Storage / UI
    MAX_GALLERY_ITEMS: int = int(os.getenv("MAX_GALLERY_ITEMS", "200"))

    # Alerts
    ALERTS_ENABLED: bool = os.getenv("ALERTS_ENABLED", "1") == "1"
    ALERT_COOLDOWN_SEC: float = float(os.getenv("ALERT_COOLDOWN_SEC", "1.0"))

    # Reports
    REPORTS_DIR: str = os.getenv("REPORTS_DIR", "reports")

    # Email
    EMAIL_ENABLED: bool = os.getenv("EMAIL_ENABLED", "0") == "1"
    SMTP_SERVER: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME: str = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    EMAIL_FROM: str = os.getenv("EMAIL_FROM", "cctv-system@company.com")
    EMAIL_TO_DEFAULT: str = os.getenv("EMAIL_TO", "")  # optional default recipients, comma-separated
    EMAIL_SUBJECT_PREFIX: str = os.getenv("EMAIL_SUBJECT_PREFIX", "PPE Compliance Report")

def get_settings() -> Settings:
    return Settings()
