"""
MIDAN Data Pipeline -- Base Extractor (Phase 2)
Abstract base class for all context-aware extractors.
Includes hardened classification engine with multi-keyword validation,
negative filtering, and priority-based conflict resolution.
"""

from abc import ABC, abstractmethod
from utils.logger import setup_logger
from config.settings import (
    LOG_FILE, BUSINESS_MODEL_RULES, INDUSTRY_RULES,
    BUSINESS_MODEL_CANONICAL, INDUSTRY_CANONICAL,
    TARGET_SEGMENT_RULES,
    RETENTION_KEYWORDS, ONBOARDING_KEYWORDS,
    MONETIZATION_KEYWORDS, COMPETITION_KEYWORDS,
)


class BaseExtractor(ABC):
    """
    Abstract base for context-aware extractors.
    Each extractor handles a specific source type and knows
    which fields to fill vs. leave empty.
    """

    def __init__(self, source_type: str):
        self.source_type = source_type
        self.logger = setup_logger(f"extractor.{source_type}", LOG_FILE)

    @abstractmethod
    def extract(self, raw_entry: dict) -> dict:
        """
        Extract structured fields from a raw data entry.

        Args:
            raw_entry: Dict from collector with raw_content, metadata, etc.

        Returns:
            Dict matching the target schema.
            Empty fields should be "" or [] -- NEVER fabricated.
        """
        pass

    def _empty_entry(self) -> dict:
        """Create an empty schema-compliant entry with Phase 2 fields."""
        return {
            "startup_name": "",
            "primary_industry": "",
            "secondary_industry": "",
            "business_model": "",
            "target_user": "",
            "target_segment": "",       # B2B / B2C
            "value_proposition": "",
            "pain_points": [],
            "adoption_barriers": [],
            "user_complaints": [],
            "success_drivers": [],
            "failure_reasons": [],
            "differentiation": "",
            "switching_cost": "",
            "competition_density": "",
            "signal_evidence": {},       # maps field_name -> list of evidence strings
            "confidence_score": 0.0,     # 0.0 - 1.0
            # Phase 2 L3 signals
            "retention_proxy": "",       # high / medium / low
            "onboarding_friction": "",   # low / medium / high
            "monetization_strength": "", # strong / moderate / weak
            "competition_intensity": "", # high / moderate / low
            "funding_stage": "",         # pre_seed / seed / series_a / etc.
            "market_context": "",        # brief market description
            "source_type": self.source_type,
            "source_url": "",
        }

    # ──────────────────────────────────────────────
    # HARDENED CLASSIFICATION ENGINE (Phase 2)
    # ──────────────────────────────────────────────

    @staticmethod
    def _kw_in_text(keyword: str, text_lower: str) -> bool:
        """
        Check if keyword exists in text.
        For short keywords (<=3 chars), use word-boundary matching
        to prevent false positives (e.g., 'ai' matching 'again').
        """
        import re
        kw = keyword.lower()
        if len(kw) <= 3:
            return bool(re.search(r'\b' + re.escape(kw) + r'\b', text_lower))
        return kw in text_lower

    def _classify_with_rules(self, text: str, rules_dict: dict) -> str:
        """
        Classify text using hardened multi-keyword rules.

        Scoring:
        - primary keyword match = 2 points
        - secondary keyword match = 1 point
        - negative keyword match = -3 points
        - Must exceed min_score threshold
        - Ties broken by priority (lower = higher priority)

        Returns normalized canonical label or "".
        """
        text_lower = text.lower()
        candidates = []

        for category, rules in rules_dict.items():
            primary = rules.get("primary", [])
            secondary = rules.get("secondary", [])
            negative = rules.get("negative", [])
            min_score = rules.get("min_score", 2)
            priority = rules.get("priority", 10)

            # Score calculation with word-boundary awareness for short keywords
            primary_hits = sum(2 for kw in primary if self._kw_in_text(kw, text_lower))
            secondary_hits = sum(1 for kw in secondary if self._kw_in_text(kw, text_lower))
            negative_hits = sum(3 for kw in negative if self._kw_in_text(kw, text_lower))

            total_score = primary_hits + secondary_hits - negative_hits

            if total_score >= min_score:
                candidates.append((category, total_score, priority))

        if not candidates:
            return ""

        # Sort by score (descending), then by priority (ascending)
        candidates.sort(key=lambda x: (-x[1], x[2]))

        winner = candidates[0][0]
        return winner

    def _classify_with_rules_multi(self, text: str, rules_dict: dict, top_n: int = 2) -> list[str]:
        """
        Classify text using hardened multi-keyword rules, returning top N matches.
        """
        text_lower = text.lower()
        candidates = []

        for category, rules in rules_dict.items():
            primary = rules.get("primary", [])
            secondary = rules.get("secondary", [])
            negative = rules.get("negative", [])
            min_score = rules.get("min_score", 2)
            priority = rules.get("priority", 10)

            primary_hits = sum(2 for kw in primary if self._kw_in_text(kw, text_lower))
            secondary_hits = sum(1 for kw in secondary if self._kw_in_text(kw, text_lower))
            negative_hits = sum(3 for kw in negative if self._kw_in_text(kw, text_lower))

            total_score = primary_hits + secondary_hits - negative_hits

            if total_score >= min_score:
                candidates.append((category, total_score, priority))

        if not candidates:
            return []

        candidates.sort(key=lambda x: (-x[1], x[2]))
        return [c[0] for c in candidates[:top_n]]

    def _classify_business_model(self, text: str) -> str:
        """Classify business model using hardened rules. Returns canonical label."""
        result = self._classify_with_rules(text, BUSINESS_MODEL_RULES)
        return result

    def _classify_industry(self, text: str) -> tuple[str, str]:
        """Classify industry using hardened rules. Returns (primary, secondary)."""
        results = self._classify_with_rules_multi(text, INDUSTRY_RULES, top_n=2)
        primary = results[0] if len(results) > 0 else ""
        secondary = results[1] if len(results) > 1 else ""
        return primary, secondary

    def _classify_target_segment(self, text: str) -> str:
        """Classify as B2B or B2C based on keyword density."""
        text_lower = text.lower()

        b2b_score = sum(1 for kw in TARGET_SEGMENT_RULES["B2B"] if kw in text_lower)
        b2c_score = sum(1 for kw in TARGET_SEGMENT_RULES["B2C"] if kw in text_lower)

        if b2b_score > b2c_score and b2b_score >= 2:
            return "B2B"
        elif b2c_score > b2b_score and b2c_score >= 2:
            return "B2C"
        elif b2b_score >= 2 and b2c_score >= 2:
            return "B2B + B2C"
        return ""

    # ──────────────────────────────────────────────
    # L3 SIGNAL EXTRACTION (Phase 2)
    # ──────────────────────────────────────────────

    def _classify_level(self, text: str, level_keywords: dict) -> tuple[str, list[str]]:
        """Generic level classifier: returns the level with highest score and evidence."""
        text_lower = text.lower()
        scores = {}
        for level, keywords in level_keywords.items():
            scores[level] = sum(1 for kw in keywords if kw in text_lower)

        if not scores:
            return "", []

        best = max(scores, key=scores.get)
        if scores[best] >= 2:
            # Extract evidence
            evidence = self._extract_matching_phrases(text, level_keywords[best], context_window=100)
            return best, evidence[:2]
        return "", []

    def _extract_retention_proxy(self, text: str) -> tuple[str, list[str]]:
        return self._classify_level(text, RETENTION_KEYWORDS)

    def _extract_onboarding_friction(self, text: str) -> tuple[str, list[str]]:
        result, evidence = self._classify_level(text, ONBOARDING_KEYWORDS)
        mapping = {"low_friction": "low", "medium_friction": "medium", "high_friction": "high"}
        return mapping.get(result, result), evidence

    def _extract_monetization_strength(self, text: str) -> tuple[str, list[str]]:
        return self._classify_level(text, MONETIZATION_KEYWORDS)

    def _extract_competition_intensity(self, text: str) -> tuple[str, list[str]]:
        return self._classify_level(text, COMPETITION_KEYWORDS)

    # ──────────────────────────────────────────────
    # LEGACY KEYWORD MATCHING (still used by extractors)
    # ──────────────────────────────────────────────

    def _keyword_match(self, text: str, keyword_dict: dict) -> str:
        """
        Find the best-matching category from a keyword dictionary.
        Returns the category with the most keyword hits.
        """
        text_lower = text.lower()
        scores = {}

        for category, keywords in keyword_dict.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > 0:
                scores[category] = score

        if not scores:
            return ""

        return max(scores, key=scores.get)

    def _keyword_match_multi(self, text: str, keyword_dict: dict, top_n: int = 3) -> list[str]:
        """
        Find multiple matching categories from a keyword dictionary.
        Returns top N categories by score.
        """
        text_lower = text.lower()
        scores = {}

        for category, keywords in keyword_dict.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > 0:
                scores[category] = score

        if not scores:
            return []

        sorted_cats = sorted(scores, key=scores.get, reverse=True)
        return sorted_cats[:top_n]

    def _extract_matching_phrases(self, text: str, keywords: list[str],
                                   context_window: int = 80) -> list[str]:
        """
        Find sentences/phrases containing any of the keywords.
        Returns a list of contextual snippets.
        """
        text_lower = text.lower()
        phrases = []
        seen = set()

        for keyword in keywords:
            idx = 0
            while True:
                pos = text_lower.find(keyword.lower(), idx)
                if pos == -1:
                    break

                # Extract surrounding context
                start = max(0, pos - context_window)
                end = min(len(text), pos + len(keyword) + context_window)

                snippet = text[start:end].strip()

                # Try to start/end at sentence boundary
                dot_before = snippet.rfind(".", 0, context_window)
                if dot_before > 0:
                    snippet = snippet[dot_before + 1:].strip()

                dot_after = snippet.find(".", context_window)
                if dot_after > 0:
                    snippet = snippet[:dot_after + 1].strip()

                # Deduplicate
                snippet_key = snippet[:50].lower()
                if snippet_key not in seen and len(snippet) > 20:
                    seen.add(snippet_key)
                    phrases.append(snippet)

                idx = pos + len(keyword)

        return phrases[:10]
