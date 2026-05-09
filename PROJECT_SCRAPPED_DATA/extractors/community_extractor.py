"""
MIDAN Data Pipeline -- Community Extractor
Extracts user sentiment, pain points, complaints, and switching behavior
from Reddit posts and comments.
"""

import re
from extractors.base_extractor import BaseExtractor
from utils.text_cleaner import clean_text, truncate
from config.settings import PAIN_KEYWORDS


# Reddit-specific complaint indicators
COMPLAINT_KEYWORDS = [
    "hate", "worst", "terrible", "horrible", "awful", "trash",
    "useless", "waste of time", "waste of money", "scam", "rip off",
    "downgrade", "worse than", "switched from", "left because",
    "cancelled", "unsubscribed", "stopped using", "moved to",
    "disappointed", "frustrating", "buggy", "crashes", "laggy",
    "overpriced", "not worth", "avoid", "don't recommend",
]

SWITCHING_KEYWORDS = [
    "switched to", "switched from", "moved to", "moved from",
    "migrated to", "migrated from", "replaced with", "alternative to",
    "better than", "instead of", "left for", "dropped for",
    "comparing", "vs", "versus",
]

POSITIVE_KEYWORDS = [
    "love", "amazing", "great", "excellent", "best", "perfect",
    "recommend", "game changer", "life saver", "essential",
    "fantastic", "impressive", "solid", "reliable", "powerful",
]


class CommunityExtractor(BaseExtractor):
    """
    Extracts user sentiment signals from Reddit/community content.

    Primary fields: user_complaints, pain_points, switching_cost
    """

    def __init__(self):
        super().__init__("community")

    def extract(self, raw_entry: dict) -> dict:
        """Extract structured fields from Reddit post content."""
        entry = self._empty_entry()
        entry["startup_name"] = raw_entry.get("startup_name", "")
        entry["source_url"] = raw_entry.get("source_url", "")
        entry["source_type"] = "community"

        raw_content = raw_entry.get("raw_content", "")
        metadata = raw_entry.get("metadata", {})

        if not raw_content:
            return entry

        text = clean_text(raw_content)

        # ── USER COMPLAINTS ──
        entry["user_complaints"] = self._extract_complaints(text)

        # ── PAIN POINTS ──
        entry["pain_points"] = self._extract_pain_points(text)

        # ── SWITCHING SIGNALS ──
        def add_signal(field_name, result_tuple):
            level, evidence = result_tuple
            entry[field_name] = level
            if level and evidence:
                entry["signal_evidence"][field_name] = evidence

        sw_level, sw_evidence = self._extract_switching_signals(text)
        add_signal("switching_cost", (sw_level, sw_evidence))

        # ── SENTIMENT-BASED SIGNALS ──
        add_signal("retention_proxy", self._infer_retention(text))
        add_signal("competition_intensity", self._infer_competition(text))

        # ── INDUSTRY / MODEL (from context if available) ──
        entry["primary_industry"], entry["secondary_industry"] = self._classify_industry(text)
        entry["business_model"] = self._classify_business_model(text)
        entry["target_segment"] = self._classify_target_segment(text)

        return entry

    def _extract_complaints(self, text: str) -> list[str]:
        """Extract user complaints from Reddit content."""
        complaints = []
        # Focus on COMMENT sections which contain user opinions
        comment_sections = re.findall(r"\[COMMENT[^\]]*\]\s*(.*?)(?=\[COMMENT|\[TITLE\]|\Z)",
                                       text, re.DOTALL)

        search_text = "\n".join(comment_sections) if comment_sections else text

        phrases = self._extract_matching_phrases(search_text, COMPLAINT_KEYWORDS,
                                                  context_window=100)
        for phrase in phrases:
            cleaned = truncate(clean_text(phrase), 200)
            if cleaned and len(cleaned) > 20:
                complaints.append(cleaned)

        return complaints[:8]

    def _extract_pain_points(self, text: str) -> list[str]:
        """Extract pain points from community content."""
        phrases = self._extract_matching_phrases(text, PAIN_KEYWORDS, context_window=100)
        clean_pains = []
        for phrase in phrases:
            cleaned = truncate(clean_text(phrase), 200)
            if cleaned and len(cleaned) > 20:
                clean_pains.append(cleaned)
        return clean_pains[:5]

    def _extract_switching_signals(self, text: str) -> tuple[str, list[str]]:
        """Extract switching behavior and alternative mentions."""
        text_lower = text.lower()

        # Detect switching mentions
        switching_count = sum(1 for kw in SWITCHING_KEYWORDS if kw in text_lower)

        level = ""
        if switching_count >= 3:
            level = "low"  # lots of switching = low switching cost
        elif switching_count >= 1:
            level = "medium"

        if level:
            evidence = self._extract_matching_phrases(text, SWITCHING_KEYWORDS, context_window=100)
            return level, evidence[:2]
        return "", []

    def _infer_retention(self, text: str) -> tuple[str, list[str]]:
        """Infer retention signal from community sentiment."""
        text_lower = text.lower()

        positive_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
        negative_count = sum(1 for kw in COMPLAINT_KEYWORDS if kw in text_lower)

        level = ""
        evidence_kws = []
        if positive_count > negative_count and positive_count >= 3:
            level = "high"
            evidence_kws = POSITIVE_KEYWORDS
        elif negative_count > positive_count and negative_count >= 3:
            level = "low"
            evidence_kws = COMPLAINT_KEYWORDS

        if level:
            evidence = self._extract_matching_phrases(text, evidence_kws, context_window=100)
            return level, evidence[:2]
        return "", []

    def _infer_competition(self, text: str) -> tuple[str, list[str]]:
        """Infer competition intensity from community mentions."""
        text_lower = text.lower()
        switching_count = sum(1 for kw in SWITCHING_KEYWORDS if kw in text_lower)

        level = ""
        if switching_count >= 3:
            level = "high"
        elif switching_count >= 1:
            level = "moderate"
            
        if level:
            evidence = self._extract_matching_phrases(text, SWITCHING_KEYWORDS, context_window=100)
            return level, evidence[:2]
        return "", []
