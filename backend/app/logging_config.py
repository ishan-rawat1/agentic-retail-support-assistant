"""Structured logging configuration."""

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers = [handler]
    # Quiet noisy third-party loggers.
    for name in ("httpx", "openai", "pinecone", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
