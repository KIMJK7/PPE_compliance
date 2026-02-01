from __future__ import annotations

import time
import threading
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Source = Union[int, str]

@dataclass
class FramePacket:
    frame_bgr: np.ndarray
    ts: float
    width: int
    height: int

class CameraReader:
    """Continuously reads frames from RTSP or webcam with auto-reconnect.

    This handles common Windows backends:
    - Webcam indices: CAP_DSHOW then CAP_MSMF
    - RTSP/URL: CAP_FFMPEG then default
    """

    def __init__(self, source: Source, width: int = 640, height: int = 360, max_fps: int = 20):
        self.source = source
        self.width = width
        self.height = height
        self.max_fps = max(1, max_fps)

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._latest: Optional[FramePacket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._release()

    def is_active(self) -> bool:
        with self._lock:
            return self._latest is not None and (time.time() - self._latest.ts) < 2.5

    def latest(self) -> Optional[FramePacket]:
        with self._lock:
            return self._latest

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None

    def _open(self) -> bool:
        self._release()
        src = self.source

        if isinstance(src, int):
            # webcam
            for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]:
                cap = cv2.VideoCapture(src, backend) if backend != 0 else cv2.VideoCapture(src)
                if cap.isOpened():
                    self._cap = cap
                    break
            if self._cap is None:
                return False
            # try set dimensions
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.max_fps)
            return True

        # URL/RTSP
        for backend in [cv2.CAP_FFMPEG, 0]:
            cap = cv2.VideoCapture(src, backend) if backend != 0 else cv2.VideoCapture(src)
            if cap.isOpened():
                self._cap = cap
                break
        if self._cap is None:
            return False
        return True

    def _loop(self) -> None:
        desired_dt = 1.0 / float(self.max_fps)
        backoff = 0.5

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                ok = self._open()
                if not ok:
                    logger.warning("Camera open failed. Retrying in %.1fs", backoff)
                    time.sleep(backoff)
                    backoff = min(5.0, backoff * 1.5)
                    continue
                logger.info("Camera opened: %s", self.source)
                backoff = 0.5

            t0 = time.time()
            ok, frame = self._cap.read()
            if not ok or frame is None:
                logger.warning("Frame read failed. Reconnecting...")
                self._release()
                time.sleep(0.2)
                continue

            # resize if requested
            if self.width and self.height:
                try:
                    frame = cv2.resize(frame, (self.width, self.height))
                except Exception:
                    pass

            h, w = frame.shape[:2]
            pkt = FramePacket(frame_bgr=frame, ts=time.time(), width=w, height=h)
            with self._lock:
                self._latest = pkt

            dt = time.time() - t0
            if dt < desired_dt:
                time.sleep(desired_dt - dt)
                
def generate_frames(reader: CameraReader):
    """Flask-compatible MJPEG generator"""
    while True:
        pkt = reader.latest()
        if pkt is None:
            time.sleep(0.05)
            continue

        frame = pkt.frame_bgr
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buffer.tobytes() +
            b'\r\n'
        )
