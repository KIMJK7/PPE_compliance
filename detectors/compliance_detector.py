from __future__ import annotations

import os
import logging
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .yolo_utils import parse_yolo_output

logger = logging.getLogger(__name__)

COMPLIANCE_CLASSES: Dict[int, str] = {
    0: "exposed_hair",
    1: "exposed_ear",
    2: "head_net",
    3: "beard_net",
    4: "exposed_beard",
}

@dataclass
class ComplianceResult:
    is_compliant: bool
    status: str
    reason: str
    detected_classes: List[str]
    confidence_scores: Dict[str, float]
    violations: List[str]
    detections: List[Dict]
    timestamp: str

    def to_dict(self) -> Dict:
        return {
            "is_compliant": self.is_compliant,
            "status": self.status,
            "reason": self.reason,
            "detected_classes": self.detected_classes,
            "confidence_scores": self.confidence_scores,
            "violations": self.violations,
            "detections": self.detections,
            "timestamp": self.timestamp,
        }

class ComplianceDetector:
    """Runs YOLO compliance model (ONNX) and applies rule-based compliance."""

    def __init__(
        self,
        model_path: str,
        model_url: Optional[str] = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ):
        self.model_path = model_path
        self.model_url = model_url
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.input_size = 640

        self._session: Optional[ort.InferenceSession] = None
        self._input_name: Optional[str] = None

    def ensure_model(self) -> None:
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        if os.path.exists(self.model_path):
            return
        if not self.model_url:
            raise FileNotFoundError(
                f"Compliance model not found at {self.model_path} and COMPLIANCE_MODEL_URL not set."
            )
        logger.info("Downloading compliance model to %s", self.model_path)
        urllib.request.urlretrieve(self.model_url, self.model_path)

    def load(self) -> None:
        self.ensure_model()
        providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(self.model_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        logger.info("✅ Compliance detector loaded (%s)", self.model_path)

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame_bgr, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        blob = np.transpose(img, (2, 0, 1))[None, ...]
        return blob

    def detect(self, frame_bgr: np.ndarray) -> ComplianceResult:
        if self._session is None or self._input_name is None:
            ts = datetime.now().isoformat(timespec="seconds")
            return ComplianceResult(
                is_compliant=False,
                status="ERROR",
                reason="Compliance model not loaded",
                detected_classes=[],
                confidence_scores={},
                violations=["model_not_loaded"],
                detections=[],
                timestamp=ts,
            )

        h, w = frame_bgr.shape[:2]
        blob = self._preprocess(frame_bgr)
        outputs = self._session.run(None, {self._input_name: blob})
        dets_raw = parse_yolo_output(
            outputs,
            input_size=self.input_size,
            orig_w=w,
            orig_h=h,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            num_classes=len(COMPLIANCE_CLASSES),
        )

        detections: List[Dict] = []
        detected_names: List[str] = []
        conf_scores: Dict[str, float] = {}

        for d in dets_raw:
            cls_id = int(d["class_id"])
            name = COMPLIANCE_CLASSES.get(cls_id, f"unknown_{cls_id}")
            conf = float(d["conf"])
            x1, y1, x2, y2 = [float(v) for v in d["xyxy"]]
            detections.append({"class_id": cls_id, "class_name": name, "conf": conf, "xyxy": [x1, y1, x2, y2]})
            detected_names.append(name)
            if name not in conf_scores or conf > conf_scores[name]:
                conf_scores[name] = conf

        is_compliant, violations, reason = self._apply_rules(detected_names)
        ts = datetime.now().isoformat(timespec="seconds")
        return ComplianceResult(
            is_compliant=is_compliant,
            status="COMPLIANT" if is_compliant else "NON-COMPLIANT",
            reason=reason,
            detected_classes=sorted(list(set(detected_names))),
            confidence_scores=conf_scores,
            violations=violations,
            detections=detections,
            timestamp=ts,
        )

    def _apply_rules(self, detected: List[str]) -> Tuple[bool, List[str], str]:
        s = set(detected)
        violations: List[str] = []

        # Non-compliance conditions
        if "exposed_hair" in s:
            violations.append("exposed_hair")
        if "exposed_beard" in s:
            violations.append("exposed_beard")
        if "exposed_ear" in s and "head_net" not in s:
            violations.append("exposed_ear_without_head_net")

        # Compliance logic
        if violations:
            return False, violations, " / ".join(violations)

        # Compliant if at least head_net present (and no exposed conditions)
        if "head_net" in s and "beard_net" in s:
            return True, [], "head_net + beard_net"
        if "head_net" in s:
            return True, [], "head_net"
        if "beard_net" in s:
            # If beard_net but no head_net, still not fully compliant for head hair.
            return False, ["missing_head_net"], "missing_head_net"

        return False, ["required_ppe_not_detected"], "required_ppe_not_detected"
