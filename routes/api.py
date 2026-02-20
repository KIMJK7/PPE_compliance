from __future__ import annotations

import time
import logging
import base64
from typing import Optional

import cv2
import numpy as np
from flask import Blueprint, jsonify, request, Response, send_file

from core.state import FrameRecord
from pipeline.processor import Pipeline, annotate_frame
from reporting.pdf_report import PDFReportGenerator
from reporting.email_sender import EmailSender

logger = logging.getLogger(__name__)

# Used for manual posting + upload endpoint filtering
MIN_FRAME_CONFIDENCE = 0.25


def make_api_bp(
    pipeline: Pipeline,
    report_gen: PDFReportGenerator,
    email_sender: Optional[EmailSender],
):
    api_bp = Blueprint("api", __name__)

    @api_bp.get("/api/frames")
    def get_frames():
        """
        Returns dashboard frames payload.

        IMPORTANT:
        - With TrackFrameStore, payload contains BOTH:
          - grouped: compliant_tracks / non_compliant_tracks
          - legacy flat: compliant / non_compliant
        So older UI/reporting won't break.
        """
        return jsonify(pipeline.store.to_payload())

    @api_bp.post("/api/frames")
    def post_frame():
        """
        Allows manually posting a frame record into the store.
        Expects JSON fields similar to FrameRecord.
        """
        data = request.get_json(force=True, silent=True) or {}

        avg_conf = float(data.get("avg_confidence") or 0.0)
        if avg_conf < MIN_FRAME_CONFIDENCE:
            return jsonify({"ok": True, "ignored": True, "reason": "low_confidence"}), 200

        rec = FrameRecord(
            timestamp=data.get("timestamp")
            or __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            reason=str(data.get("reason") or ""),
            classes=list(data.get("classes") or []),
            avg_confidence=float(data.get("avg_confidence") or 0.0),
            image_data=str(data.get("image_data") or ""),
            detections=list(data.get("detections") or []),
            is_compliant=True if str(data.get("type") or "") == "compliant" else False,
            # Optional. If not provided, defaults to "unknown" in FrameRecord.
            track_id=str(data.get("track_id") or "manual"),
        )
        pipeline.store.add(rec)
        return jsonify({"ok": True})

    @api_bp.post("/api/frames/clear")
    def clear_frames():
        pipeline.store.clear()
        return jsonify({"ok": True})

    @api_bp.get("/api/cctv/status")
    def cctv_status():
        pkt = pipeline.camera.latest()
        dims = None
        if pkt is not None:
            dims = {"width": pkt.width, "height": pkt.height}

        return jsonify(
            {
                "stream_active": pipeline.camera.is_active(),
                "frame_dimensions": dims,
                "person_detection": {
                    "total_frames_processed": pipeline.stats.total_frames_processed,
                    "frames_with_person": pipeline.stats.frames_with_person,
                    "frames_without_person": pipeline.stats.frames_without_person,
                    "person_detection_rate": pipeline.stats.person_detection_rate,
                },
                "compliance": {
                    "compliant_frames": pipeline.stats.compliant_frames,
                    "non_compliant_frames": pipeline.stats.non_compliant_frames,
                    "compliance_rate": pipeline.stats.compliance_rate,
                },
            }
        )

    @api_bp.get("/api/cctv/compliance")
    def latest_compliance():
        res = pipeline.latest_result()
        if res is None:
            return jsonify({"error": "no compliance result yet"}), 503
        return jsonify(res.to_dict())

    @api_bp.get("/video_feed")
    def video_feed():
        """MJPEG stream endpoint (capped FPS + lower JPEG quality for low latency)."""
        STREAM_FPS = 12
        interval = 1.0 / float(STREAM_FPS)
        last = 0.0

        def gen():
            nonlocal last
            while True:
                now = time.time()
                if now - last < interval:
                    time.sleep(0.005)
                    continue
                last = now

                frame = pipeline.latest_annotated()
                if frame is None:
                    pkt = pipeline.camera.latest()
                    frame = None if pkt is None else pkt.frame_bgr

                if frame is None:
                    time.sleep(0.05)
                    continue

                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if not ok:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @api_bp.get("/api/cctv/video_feed")
    def cctv_video_feed_alias():
        return video_feed()

    @api_bp.post("/api/report/generate")
    def generate_report():
        data = request.get_json(silent=True) or {}
        email = (data.get("email") or "").strip()

        payload = pipeline.store.to_payload()
        pdf_path = report_gen.generate(payload)

        if email:
            if email_sender is None:
                return jsonify({"success": False, "error": "Email is disabled or SMTP not configured"}), 400
            try:
                recipients = [e.strip() for e in email.split(",") if e.strip()]
                count = int(payload.get("total", 0))
                email_sender.send_pdf(pdf_path, recipients, count)
                return jsonify({"success": True, "pdf_path": pdf_path})
            except Exception as e:
                logger.exception("Email send failed")
                return jsonify({"success": False, "error": str(e)}), 500

        return send_file(pdf_path, as_attachment=True, download_name="compliance_report.pdf")

    @api_bp.post("/api/upload/frame")
    def upload_frame_detect():
        if "file" not in request.files:
            return jsonify({"success": False, "error": "Missing file field"}), 400

        data = request.files["file"].read()
        if not data:
            return jsonify({"success": False, "error": "Empty file"}), 400

        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"success": False, "error": "Failed to decode image"}), 400

        # Person/head gating
        has_person = True
        if pipeline.person_detector is not None:
            has_person = bool(pipeline.person_detector.has_person(frame))

        if not has_person:
            return jsonify(
                {
                    "success": True,
                    "status": "NO_PERSON",
                    "reason": "no_person_detected",
                    "classes": [],
                    "detections": [],
                    "avg_confidence": 0.0,
                    "annotated_image_data": None,
                }
            )

        result = pipeline.compliance_detector.detect(frame)

        if result.status == "NO_PERSON":
            return jsonify(
                {
                    "success": True,
                    "status": "NO_PERSON",
                    "reason": result.reason,
                    "classes": [],
                    "detections": [],
                    "avg_confidence": 0.0,
                    "annotated_image_data": None,
                }
            )

        avg_conf = float(np.mean([d["conf"] for d in result.detections])) if result.detections else 0.0

        if avg_conf < MIN_FRAME_CONFIDENCE:
            return jsonify(
                {
                    "success": True,
                    "ignored": True,
                    "status": "IGNORED",
                    "reason": "low_confidence",
                    "avg_confidence": avg_conf,
                    "classes": [],
                    "detections": [],
                }
            )

        annotated = annotate_frame(frame, result)
        ok, jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            return jsonify({"success": False, "error": "Failed to encode annotated frame"}), 500

        b64 = base64.b64encode(jpg.tobytes()).decode("utf-8")

        return jsonify(
            {
                "success": True,
                "status": result.status,
                "reason": result.reason,
                "classes": result.detected_classes,
                "detections": result.detections,
                "avg_confidence": avg_conf,
                "annotated_image_data": f"data:image/jpeg;base64,{b64}",
            }
        )

    return api_bp
