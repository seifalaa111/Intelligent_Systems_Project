"""
MIDAN Data Pipeline -- Directory Extractor
Extracts product positioning, category, and differentiation signals
from Product Hunt and similar directory pages.
"""

import re
from extractors.base_extractor import BaseExtractor
from utils.text_cleaner import clean_text, truncate


class DirectoryExtractor(BaseExtractor):
    """
    Extracts structured intelligence from product directory pages.

    Primary fields: value_proposition, industry, differentiation
    """

    def __init__(self):
        super().__init__("directory")

    def extract(self, raw_entry: dict) -> dict:
        """Extract structured fields from directory content."""
        entry = self._empty_entry()
        entry["startup_name"] = raw_entry.get("startup_name", "")
        entry["source_url"] = raw_entry.get("source_url", "")
        entry["source_type"] = "directory"

        raw_content = raw_entry.get("raw_content", "")
        metadata = raw_entry.get("metadata", {})

        if not raw_content:
            return entry

        text = clean_text(raw_content)

        # ── VALUE PROPOSITION (from tagline or meta) ──
        tagline = metadata.get("tagline", "")
        if tagline and 10 < len(tagline) < 300:
            entry["value_proposition"] = tagline
        else:
            # Extract from content
            meta_tags = metadata.get("meta_tags", {})
            if meta_tags.get("og_description"):
                entry["value_proposition"] = meta_tags["og_description"][:300]

        # ── INDUSTRY & MODEL ──
        search_text = f"{tagline} {text[:2000]}"
        entry["primary_industry"], entry["secondary_industry"] = self._classify_industry(search_text)
        entry["business_model"] = self._classify_business_model(search_text)
        entry["target_segment"] = self._classify_target_segment(search_text)

        # ── TOPICS / CATEGORIES ──
        topics = metadata.get("topics", [])
        if topics:
            entry["differentiation"] = f"Product Hunt categories: {', '.join(topics[:5])}"

        # ── COMPETITION (from "alternatives" or "vs" mentions) ──
        comp_level, comp_ev = self._extract_competition_from_directory(text)
        entry["competition_intensity"] = comp_level
        if comp_level and comp_ev:
            entry["signal_evidence"]["competition_intensity"] = comp_ev

        return entry

    def _extract_competition_from_directory(self, text: str) -> tuple[str, list[str]]:
        """Infer competition from directory page content."""
        text_lower = text.lower()

        alt_signals = ["alternative", "alternatives", "compare", "vs ",
                       "competitor", "similar to", "like"]

        alt_count = sum(1 for s in alt_signals if s in text_lower)

        level = ""
        if alt_count >= 3:
            level = "high"
        elif alt_count >= 1:
            level = "moderate"
            
        if level:
            ev = self._extract_matching_phrases(text, alt_signals, context_window=100)
            return level, ev[:2]
        return "", []
