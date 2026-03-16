# core/config.py
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from dotenv import load_dotenv


# ──────────────────────────────────────────────
# PyInstaller bundle path helper
# ──────────────────────────────────────────────
def resource_path(relative_path: str) -> str:
    """
    Returns the absolute path to a resource.
    - When running as a PyInstaller .exe → uses sys._MEIPASS (temp bundle dir)
    - When running normally             → uses the directory of this file's project root
    """
    if getattr(sys, "frozen", False):
        # Running inside PyInstaller bundle
        base_path = sys._MEIPASS
    else:
        # Running normally — go up one level from core/ to project root
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base_path, relative_path)


# ──────────────────────────────────────────────
# Load .env from correct location
# ──────────────────────────────────────────────
_dotenv_path = resource_path(".env")
if os.path.exists(_dotenv_path):
    load_dotenv(_dotenv_path)
else:
    load_dotenv()  # fallback: search default locations


# ──────────────────────────────────────────────
# Runtime base directory
# Used to resolve writable dirs (reports/, models/)
# ──────────────────────────────────────────────
def runtime_base() -> str:
    """
    Returns the writable base directory at runtime.
    - PyInstaller .exe → folder containing the .exe
    - Normal run       → project root
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def writable_path(relative_path: str) -> str:
    """
    Returns an absolute path under the runtime base directory.
    Use this for WRITABLE paths like reports/ and models/
    so they sit next to the .exe, not inside the read-only bundle.
    """
    path = os.path.join(runtime_base(), relative_path)
    os.makedirs(path, exist_ok=True)
    return path


# ──────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class Settings:

    # ── Flask ──────────────────────────────────
    FLASK_HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT: int = int(os.getenv("FLASK_PORT", "5000"))
    DEBUG: bool = os.getenv("FLASK_DEBUG", "0") == "1"

    # ── Video source ───────────────────────────
    # Use ONE of:
    #   RTSP_URL   → network camera / CCTV
    #   CAMERA_INDEX → USB webcam (0, 1, 2 …)
    RTSP_URL: str | None = os.getenv("RTSP_URL") or None

    # Resolve CAMERA_INDEX: -1 means "not set" → None
    CAMERA_INDEX: int | None = (
        None
        if int(os.getenv("CAMERA_INDEX", "-1")) == -1
        else int(os.getenv("CAMERA_INDEX", "-1"))
    )

    FRAME_WIDTH:  int = int(os.getenv("FRAME_WIDTH",  "640"))
    FRAME_HEIGHT: int = int(os.getenv("FRAME_HEIGHT", "360"))
    MAX_FPS:      int = int(os.getenv("MAX_FPS",      "20"))

    # ── Person-gating model (YOLOv5n ONNX) ────
    MODEL_DIR: str = writable_path(os.getenv("MODEL_DIR", "models"))

    PERSON_MODEL_URL: str = os.getenv(
        "PERSON_MODEL_URL",
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx",
    )
    PERSON_MODEL_FILENAME: str = os.getenv("PERSON_MODEL_FILENAME", "yolov5n.onnx")
    PERSON_CONF_THRES: float = float(os.getenv("PERSON_CONF_THRES", "0.50"))

    # ── Compliance model (your trained ONNX) ───
    COMPLIANCE_MODEL_PATH: str = os.path.join(
        writable_path(os.getenv("MODEL_DIR", "models")),
        os.path.basename(os.getenv("COMPLIANCE_MODEL_PATH", "models/best.onnx")),
    )
    COMPLIANCE_MODEL_URL: str | None = os.getenv("COMPLIANCE_MODEL_URL") or None
    COMPLIANCE_CONF_THRES: float = float(os.getenv("COMPLIANCE_CONF_THRES", "0.25"))
    COMPLIANCE_IOU_THRES:  float = float(os.getenv("COMPLIANCE_IOU_THRES",  "0.45"))

    # ── Storage / UI ───────────────────────────
    MAX_GALLERY_ITEMS: int = int(os.getenv("MAX_GALLERY_ITEMS", "200"))

    # ── Alerts ─────────────────────────────────
    ALERTS_ENABLED:     bool  = os.getenv("ALERTS_ENABLED",    "1")   == "1"
    ALERT_COOLDOWN_SEC: float = float(os.getenv("ALERT_COOLDOWN_SEC", "1.0"))

    # ── Reports ────────────────────────────────
    REPORTS_DIR: str = writable_path(os.getenv("REPORTS_DIR", "reports"))

    # ── Email ──────────────────────────────────
    EMAIL_ENABLED:        bool  = os.getenv("EMAIL_ENABLED", "0") == "1"
    SMTP_SERVER:          str   = os.getenv("SMTP_SERVER",   "smtp.gmail.com")
    SMTP_PORT:            int   = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME:        str   = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD:        str   = os.getenv("SMTP_PASSWORD", "")
    EMAIL_FROM:           str   = os.getenv("EMAIL_FROM",    "cctv-system@company.com")
    EMAIL_TO_DEFAULT:     str   = os.getenv("EMAIL_TO",      "")
    EMAIL_SUBJECT_PREFIX: str   = os.getenv("EMAIL_SUBJECT_PREFIX", "PPE Compliance Report")


# ──────────────────────────────────────────────
# Singleton accessor
# ──────────────────────────────────────────────
def get_settings() -> Settings:
    return Settings()
