from __future__ import annotations

import os
import logging

from flask import Flask, jsonify
from flask_cors import CORS

from core.config import get_settings
from core.logging_config import setup_logging
from camera.streamer import CameraReader, generate_frames
from detectors.person_detector import PersonDetector
from detectors.compliance_detector import ComplianceDetector
from core.state import FrameStore
from alerts.beeper import Beeper
from pipeline.processor import Pipeline
from reporting.pdf_report import PDFReportGenerator
from reporting.email_sender import EmailSender
from routes.pages import pages_bp
from routes.api import make_api_bp

def create_app() -> Flask:
    setup_logging()
    logger = logging.getLogger("app")
    s = get_settings()

    app = Flask(__name__, template_folder="templates", static_folder="static")
    CORS(app)

    # Choose video source
    if s.RTSP_URL:
        source = s.RTSP_URL
        logger.info("Using RTSP_URL source")
    elif s.CAMERA_INDEX is not None:
        source = int(s.CAMERA_INDEX)
        logger.info("Using CAMERA_INDEX source: %s", source)
    else:
        # default to webcam 0 to avoid 'nothing works' setups
        source = 0
        logger.warning("No RTSP_URL/CAMERA_INDEX set. Defaulting to webcam index 0.")

    camera = CameraReader(source=source, width=s.FRAME_WIDTH, height=s.FRAME_HEIGHT, max_fps=s.MAX_FPS)

    # Models
    person = None
    try:
        person = PersonDetector(
            model_dir=s.MODEL_DIR,
            model_url=s.PERSON_MODEL_URL,
            filename=s.PERSON_MODEL_FILENAME,
            conf_thres=s.PERSON_CONF_THRES,
        )
        person.load()
    except Exception as e:
        logger.exception("Person detector failed to load. Continuing without person gating (fail-open). Error: %s", e)
        person = None

    compliance = ComplianceDetector(
        model_path=s.COMPLIANCE_MODEL_PATH,
        model_url=s.COMPLIANCE_MODEL_URL,
        conf_thres=s.COMPLIANCE_CONF_THRES,
        iou_thres=s.COMPLIANCE_IOU_THRES,
    )
    compliance.load()

    store = FrameStore(top_k=2)
    beeper = Beeper(cooldown_sec=s.ALERT_COOLDOWN_SEC, enabled=s.ALERTS_ENABLED)

    pipeline = Pipeline(camera=camera, person_detector=person, compliance_detector=compliance, store=store, beeper=beeper)
    pipeline.start()

    report_gen = PDFReportGenerator(reports_dir=s.REPORTS_DIR)

    email_sender = None
    if s.EMAIL_ENABLED:
        email_sender = EmailSender(
            smtp_server=s.SMTP_SERVER,
            smtp_port=s.SMTP_PORT,
            username=s.SMTP_USERNAME,
            password=s.SMTP_PASSWORD,
            from_email=s.EMAIL_FROM,
            subject_prefix=s.EMAIL_SUBJECT_PREFIX,
        )

    # Blueprints
    app.register_blueprint(pages_bp)
    app.register_blueprint(make_api_bp(pipeline, report_gen, email_sender))

    # Attach pipeline to app for debugging / interactive use
    app.config["PIPELINE"] = pipeline

    @app.get("/health")
    def health():
        return jsonify({"ok": True})

    return app

if __name__ == "__main__":
    app = create_app()
    s = get_settings()
    app.run(host=s.FLASK_HOST, port=s.FLASK_PORT, debug=s.DEBUG, threaded=True)
