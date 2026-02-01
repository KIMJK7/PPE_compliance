from __future__ import annotations

import time
import threading
import platform
import logging

logger = logging.getLogger(__name__)

class Beeper:
    """Windows-friendly beeper with cooldown.

    Requirement:
    - Warning beep: single 800Hz tone
    - Critical beep: double 1000Hz tone
    - 1-second cooldown
    """

    def __init__(self, cooldown_sec: float = 1.0, enabled: bool = True):
        self.cooldown_sec = max(0.0, cooldown_sec)
        self.enabled = enabled
        self._lock = threading.Lock()
        self._last_beep_ts = 0.0

    def _cooldown_ok(self) -> bool:
        return (time.time() - self._last_beep_ts) >= self.cooldown_sec

    def beep_warning(self) -> None:
        self._beep(kind="warning")

    def beep_critical(self) -> None:
        self._beep(kind="critical")

    def _beep(self, kind: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            if not self._cooldown_ok():  # debounce
                return
            self._last_beep_ts = time.time()

        # Execute beep outside the lock
        try:
            system = platform.system()
            if system == "Windows":
                import winsound  # type: ignore

                if kind == "critical":
                    winsound.Beep(1000, 120)
                    time.sleep(0.15)
                    winsound.Beep(1000, 120)
                else:
                    winsound.Beep(800, 200)
            else:
                # Best-effort fallback: no-op, but keep logs.
                logger.info("Beeper called on non-Windows OS (%s). Skipping.", system)
        except Exception as e:
            logger.exception("Failed to beep: %s", e)
