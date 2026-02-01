from __future__ import annotations

import io
import logging
from typing import Any, Dict, Optional, List

import cv2
import numpy as np
from flask import Blueprint, jsonify, request, Response, send_file

from core.state import FrameRecord
from pipeline.processor import Pipeline
from reporting.pdf_report import PDFReportGenerator
from reporting.email_sender import EmailSender
from flask import Response
from camera.streamer import CameraReader, generate_frames

logger = logging.getLogger(__name__)
MIN_FRAME_CONFIDENCE = 0.25

def make_api_bp(pipeline: Pipeline, report_gen: PDFReportGenerator, email_sender: Optional[EmailSender]):
    api_bp = Blueprint("api", __name__)

    @api_bp.get("/api/frames")
    def get_frames():
        return jsonify(pipeline.store.to_payload())

    @api_bp.post("/api/frames")
    def post_frame():
        """Compatibility endpoint (used by client-side JS).
        Payload:
          {type:'compliant'|'non_compliant', reason, classes, image_data, avg_confidence}
        """
        data = request.get_json(force=True, silent=True) or {}
        # ✅ Ignore low-confidence frames completely
        avg_conf = float(data.get("avg_confidence") or 0.0)
        if avg_conf < MIN_FRAME_CONFIDENCE:
            return jsonify({"ok": True, "ignored": True, "reason": "low_confidence"}), 200

        rec = FrameRecord(
            timestamp=data.get("timestamp") or __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            reason=str(data.get("reason") or ""),
            classes=list(data.get("classes") or []),
            avg_confidence=float(data.get("avg_confidence") or 0.0),
            image_data=str(data.get("image_data") or ""),
            detections=list(data.get("detections") or []),
            is_compliant=str(data.get("type") or "") == "compliant",
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
        return jsonify({
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
            }
        })

    @api_bp.get("/api/cctv/compliance")
    def latest_compliance():
        res = pipeline.latest_result()
        if res is None:
            return jsonify({"error": "no compliance result yet"}), 503
        return jsonify(res.to_dict())

    @api_bp.get("/video_feed")
    def video_feed():
        """MJPEG stream endpoint."""
        def gen():
            while True:
                frame = pipeline.latest_annotated()
                if frame is None:
                    # if no annotated yet, show raw frame if available
                    pkt = pipeline.camera.latest()
                    frame = None if pkt is None else pkt.frame_bgr
                if frame is None:
                    import time
                    time.sleep(0.05)
                    continue
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ok:
                    continue
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @api_bp.get("/api/cctv/video_feed")
    def cctv_video_feed_alias():
        return video_feed()

    @api_bp.post("/api/report/generate")
    def generate_report():
        """Dashboard expects:
        - If body contains {email: '...'} -> return JSON with success and email send status
        - Else -> return PDF blob directly
        """
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

        # return PDF for download
        return send_file(pdf_path, as_attachment=True, download_name="compliance_report.pdf")
    @api_bp.post("/api/upload/frame")
    def upload_frame_detect():
        """
        Accepts a single image/frame and runs the SAME pipeline detectors used in live CCTV.
        Form-data:
        file: image/jpeg OR image/png
        Returns:
        {
            status, reason, classes, detections, avg_confidence,
            annotated_image_data (base64 jpg)
        }
        """
        if "file" not in request.files:
            return jsonify({"success": False, "error": "Missing file field"}), 400

        f = request.files["file"]
        data = f.read()
        if not data:
            return jsonify({"success": False, "error": "Empty file"}), 400

        npbuf = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"success": False, "error": "Failed to decode image"}), 400

        # Run same pipeline logic on this frame
        result, annotated = pipeline.run_on_external_frame(frame)
        # ✅ If upload detection confidence is low, ignore it (frontend won't show/save it)
        if float(result.get("avg_confidence", 0.0)) < MIN_FRAME_CONFIDENCE:
            return jsonify({
                "success": True,
                "ignored": True,
                "reason": "low_confidence",
                "status": "IGNORED",
                "avg_confidence": float(result.get("avg_confidence", 0.0)),
                "classes": [],
                "detections": [],
            }), 200


        ok, jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return jsonify({"success": False, "error": "Failed to encode annotated frame"}), 500

        import base64
        b64 = base64.b64encode(jpg.tobytes()).decode("utf-8")

        return jsonify({
            "success": True,
            "status": result["status"],
            "reason": result["reason"],
            "classes": result["classes"],
            "detections": result["detections"],
            "avg_confidence": result["avg_confidence"],
            "annotated_image_data": f"data:image/jpeg;base64,{b64}",
        })


    return api_bp
