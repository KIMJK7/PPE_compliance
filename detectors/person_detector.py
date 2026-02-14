from __future__ import annotations

import os
import logging
import urllib.request
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from .yolo_utils import parse_yolo_output

logger = logging.getLogger(__name__)

class PersonDetector:
    """Lightweight person-gating detector using YOLOv5n ONNX.

    We only need to know if a person is present (class_id=0 on COCO for YOLOv5).
    """

    def __init__(self, model_dir: str, model_url: str, filename: str = "yolov5n.onnx", conf_thres: float = 0.5):
        self.model_dir = model_dir
        self.model_url = model_url
        self.filename = filename
        self.conf_thres = conf_thres

        self.input_size = 640
        self._session: Optional[ort.InferenceSession] = None
        self._input_name: Optional[str] = None

    @property
    def model_path(self) -> str:
        return os.path.join(self.model_dir, self.filename)

    def ensure_model(self) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        if os.path.exists(self.model_path):
            return
        logger.info("Downloading person model to %s", self.model_path)
        urllib.request.urlretrieve(self.model_url, self.model_path)

    def load(self) -> None:
        self.ensure_model()
        providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(self.model_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        logger.info("✅ Person detector loaded (%s)", self.model_path)

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame_bgr, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float16) / 255.0
        blob = np.transpose(img, (2, 0, 1))[None, ...]
        return blob

    def has_person(self, frame_bgr: np.ndarray) -> bool:
        if self._session is None or self._input_name is None:
            return True  # fail-open
        try:
            h, w = frame_bgr.shape[:2]
            blob = self._preprocess(frame_bgr)
            blob = blob.astype(np.float32) 
            outputs = self._session.run(None, {self._input_name: blob})
            dets = parse_yolo_output(outputs, self.input_size, w, h, self.conf_thres, 0.45)
            # COCO person class is 0
            return any(d["class_id"] == 0 for d in dets)
        except Exception as e:
            logger.exception("Person detection failed (fail-open): %s", e)
            return True


