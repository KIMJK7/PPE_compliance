import logging
import os

def setup_logging(level: str | None = None) -> None:
    lvl = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
