"""
MIDAN Data Pipeline - Structured Logger
Uses Rich for beautiful console output + file logging.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Force UTF-8 on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Custom theme for pipeline stages
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "stage": "bold magenta",
})

console = Console(theme=custom_theme)


def setup_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """Create a logger with Rich console handler and optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # Rich console handler
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(logging.INFO)
    logger.addHandler(rich_handler)

    # File handler (if log_file provided)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def log_stage(logger: logging.Logger, stage: str, message: str):
    """Log a pipeline stage transition."""
    logger.info(f"[stage]>> {stage}[/stage] -- {message}")


def log_success(logger: logging.Logger, message: str):
    """Log a success event."""
    logger.info(f"[success][OK][/success] {message}")


def log_error(logger: logging.Logger, message: str):
    """Log an error event."""
    logger.error(f"[error][FAIL][/error] {message}")
