"""
MIDAN Data Pipeline — Base Collector
Abstract base class for all data collectors.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime, timezone
from utils.logger import setup_logger, log_stage, log_success, log_error
from config.settings import RAW_DIR, LOG_FILE


class BaseCollector(ABC):
    """
    Abstract base class for data collectors.
    Each collector fetches raw content from a specific source type
    and saves it to data/raw/{source_type}/.
    """

    def __init__(self, source_type: str):
        self.source_type = source_type
        self.output_dir = RAW_DIR / source_type
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(f"collector.{source_type}", LOG_FILE)
        self.collected_count = 0
        self.error_count = 0

    @abstractmethod
    def collect(self, targets: list[dict] | None = None) -> list[dict]:
        """
        Collect raw data from the source.

        Args:
            targets: Optional list of target startup dicts with at minimum
                     {"name": "...", "url": "..."} or source-specific fields.

        Returns:
            List of raw data dicts with at minimum:
            {
                "startup_name": str,
                "source_type": str,
                "source_url": str,
                "raw_content": str,
                "collected_at": str (ISO timestamp),
                "metadata": dict (source-specific extra fields),
            }
        """
        pass

    def save_raw(self, data: dict, filename: str | None = None):
        """Save a single raw data entry to disk."""
        if not filename:
            safe_name = data.get("startup_name", "unknown")
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in safe_name)
            filename = f"{safe_name}.json"

        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.logger.debug(f"Saved raw data: {filepath}")

    def save_all_raw(self, data_list: list[dict]):
        """Save all collected raw data entries."""
        for data in data_list:
            self.save_raw(data)

    def _make_raw_entry(self, startup_name: str, source_url: str,
                        raw_content: str, metadata: dict | None = None) -> dict:
        """Create a standardized raw data entry."""
        return {
            "startup_name": startup_name,
            "source_type": self.source_type,
            "source_url": source_url,
            "raw_content": raw_content,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

    def report(self) -> dict:
        """Return collection stats."""
        return {
            "source_type": self.source_type,
            "collected": self.collected_count,
            "errors": self.error_count,
        }
