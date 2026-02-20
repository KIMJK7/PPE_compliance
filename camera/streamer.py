from __future__ import annotations

import time
import threading
import logging
from dataclasses import dataclass
from typing import Optional, Union

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

    - Webcam: CAP_DSHOW -> CAP_MSMF -> default
    - RTSP/URL: CAP_FFMPEG -> default
    """

    def __init__(self, source: Source, width: int = 640, height: int = 360, max_fps: int = 20):
        self.source = source
        self.width = width
        self.height = height
        self.max_fps = max(1, int(max_fps))

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

        # Webcam
        if isinstance(src, int):
            for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]:
                cap = cv2.VideoCapture(src, backend) if backend != 0 else cv2.VideoCapture(src)
                if cap.isOpened():
                    self._cap = cap
                    break
            if self._cap is None:
                return False

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.max_fps)
            return True

        # RTSP/URL
        for backend in [cv2.CAP_FFMPEG, 0]:
            cap = cv2.VideoCapture(src, backend) if backend != 0 else cv2.VideoCapture(src)
            if cap.isOpened():
                self._cap = cap
                break

        if self._cap is None:
            return False

        # Critical for low latency RTSP
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        try:
            self._cap.set(cv2.CAP_PROP_FPS, self.max_fps)
        except Exception:
            pass

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

            if self.width and self.height:
                try:
                    frame = cv2.resize(frame, (self.width, self.height))
                except Exception:
                    pass

            h, w = frame.shape[:2]
            pkt = FramePacket(frame_bgr=frame, ts=time.time(), width=w, height=h)

            with self._lock:
                self._latest = pkt

            # Webcam: pacing is okay
            # RTSP: do NOT sleep, drain continuously to avoid latency buildup
            dt = time.time() - t0
            if dt < desired_dt and isinstance(self.source, int):
                time.sleep(desired_dt - dt)


def generate_frames(reader: CameraReader, fps: int = 12):
    """Flask MJPEG generator — capped FPS + smaller JPEG for speed."""
    interval = 1.0 / float(max(1, int(fps)))
    last = 0.0

    while True:
        now = time.time()
        if now - last < interval:
            time.sleep(0.005)
            continue
        last = now

        pkt = reader.latest()
        if pkt is None:
            time.sleep(0.05)
            continue

        ok, buf = cv2.imencode(".jpg", pkt.frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buf.tobytes() +
            b"\r\n"
        )
