"""
MIDAN Data Pipeline -- Source Classifier (Phase 2)
Routes raw data entries to the correct extractor based on source type.
"""

from extractors.website_extractor import WebsiteExtractor
from extractors.insight_extractor import InsightExtractor
from extractors.community_extractor import CommunityExtractor
from extractors.directory_extractor import DirectoryExtractor


# Source type to extractor mapping
_EXTRACTOR_MAP = {
    "website": WebsiteExtractor,
    "insight": InsightExtractor,
    "community": CommunityExtractor,
    "directory": DirectoryExtractor,
}

# Cached extractor instances
_extractor_instances = {}


def classify_source(raw_entry: dict) -> str:
    """
    Classify a raw data entry and return the appropriate extractor type.

    Routing rules:
    - entries from website collector -> website extractor
    - entries from failory/yc/insight -> insight extractor
    - entries from reddit -> community extractor
    - entries from Product Hunt -> directory extractor
    """
    source_type = raw_entry.get("source_type", "")

    # Direct match
    if source_type in _EXTRACTOR_MAP:
        return source_type

    # Infer from source URL
    source_url = raw_entry.get("source_url", "").lower()

    if "reddit.com" in source_url:
        return "community"
    elif "producthunt.com" in source_url:
        return "directory"
    elif "failory.com" in source_url:
        return "insight"
    elif "ycombinator.com" in source_url:
        return "insight"

    # Infer from metadata
    metadata = raw_entry.get("metadata", {})
    data_source = metadata.get("data_source", "")

    if "reddit" in data_source:
        return "community"
    elif "producthunt" in data_source:
        return "directory"
    elif "yc" in data_source:
        return "insight"
    elif "failory" in data_source:
        return "insight"

    # Default to website if we have raw_content with HTML-like structure
    raw_content = raw_entry.get("raw_content", "")
    if "<" in raw_content and ">" in raw_content:
        return "website"

    return "insight"


def get_available_extractors() -> dict:
    """
    Return a dict of all available extractor instances (cached).
    """
    global _extractor_instances
    if not _extractor_instances:
        _extractor_instances = {
            name: cls()
            for name, cls in _EXTRACTOR_MAP.items()
        }
    return _extractor_instances
