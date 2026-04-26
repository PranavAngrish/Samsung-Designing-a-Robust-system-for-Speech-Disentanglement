"""Structured JSON logging for training and eval runs."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that emits JSON lines to stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def log_metrics(logger: logging.Logger, step: int, metrics: dict[str, float]) -> None:
    """Log a metrics dict at INFO level with step number."""
    logger.info(json.dumps({"step": step, **metrics}))
