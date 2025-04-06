# app/core/logging_config.py
import logging
from .config import settings

def setup_logging():
    logging.basicConfig(
        level=settings.LOG_LEVEL.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Optionally disable noisy library logs
    # logging.getLogger("uvicorn.access").setLevel(logging.WARNING)