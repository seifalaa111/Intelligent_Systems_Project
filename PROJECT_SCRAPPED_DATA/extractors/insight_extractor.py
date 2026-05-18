"""
MIDAN Data Pipeline -- Insight Extractor (Phase 2)
Rule-based extraction of failure reasons, success drivers, and growth challenges
from Failory case studies and YC company data.
Enhanced with funding stage detection, market context, and L3 signals.
"""

import re
from extractors.base_extractor import BaseExtractor
from utils.text_cleaner import clean_text, truncate
from config.settings import (
    FAILURE_KEYWORDS, SUCCESS_KEYWORDS, PAIN_KEYWORDS,
    FUNDING_STAGE_KEYWORDS,
)

# we import the model-assisted extractor here so L3 signals can upgrade rule-based results
try:
    from extractors.model_signal_extractor import get_model_signal_extractor
    _MODEL_EXTRACTOR = get_model_signal_extractor()
except Exception:
    _MODEL_EXTRACTOR = None


class InsightExtractor(BaseExtractor):
    """
    Extracts structured intelligence from insight sources (Failory, YC).

    Primary fields: failure_reasons, success_drivers, industry, business_model
    Phase 2 additions: funding_stage, market_context, L3 signals
    """

    def __init__(self):
        super().__init__("insight")

    def extract(self, raw_entry: dict) -> dict:
        """Extract structured fields from insight source content."""
        entry = self._empty_entry()
        entry["startup_name"] = raw_entry.get("startup_name", "")
        entry["source_url"] = raw_entry.get("source_url", "")
        entry["source_type"] = "insight"

        raw_content = raw_entry.get("raw_content", "")
        metadata = raw_entry.get("metadata", {})

        if not raw_content:
            return entry

        text = clean_text(raw_content)

        # we inspect the source URL to pick the right extraction path
        source_url = raw_entry.get("source_url", "")
        is_failory = "failory" in source_url.lower()
        is_yc = "ycombinator" in source_url.lower()

        # ── CORE CLASSIFICATION — we run industry and model detection before signals ──
        if is_yc and metadata.get("industry"):
            entry["primary_industry"] = metadata["industry"]
            entry["secondary_industry"] = ""
        else:
            entry["primary_industry"], entry["secondary_industry"] = self._classify_industry(text)

        entry["business_model"] = self._extract_business_model(text, metadata)
        entry["target_user"] = self._extract_target_user(text, metadata)
        entry["target_segment"] = self._classify_target_segment(text)
        entry["value_proposition"] = self._extract_value_proposition(text, metadata)

        # ── FAILURE / SUCCESS SIGNALS — we branch by source so Failory gets richer extraction ──
        if is_failory:
            entry["failure_reasons"] = self._extract_failure_reasons_failory(text)
            entry["funding_stage"] = self._extract_funding_stage(text)
            entry["market_context"] = self._extract_market_context(text)
        else:
            entry["failure_reasons"] = self._extract_failure_reasons_generic(text)

        entry["success_drivers"] = self._extract_success_drivers(text, metadata)
        entry["pain_points"] = self._extract_pain_points(text)

        # ── L3 SIGNALS ──
        # we start with the rule-based baseline so there is always a fallback result
        def add_signal(field_name, result_tuple):
            level, evidence = result_tuple
            entry[field_name] = level
            if level and evidence:
                entry["signal_evidence"][field_name] = evidence

        # we write competition density to two fields for backward compatibility
        comp_level, comp_evidence = self._extract_competition_intensity(text)
        entry["competition_density"] = comp_level
        add_signal("competition_intensity", (comp_level, comp_evidence))

        entry["differentiation"] = self._extract_differentiation(text)
        add_signal("retention_proxy", self._extract_retention_proxy(text))
        add_signal("monetization_strength", self._extract_monetization_strength(text))

        # we let model results override rule-based when available, respecting the strict boundary:
        # ModelSignalExtractor reads ONLY raw text — it never sees
        # decision_analysis, system_patterns, or confidence_score.
        if _MODEL_EXTRACTOR is not None:
            model_vals, model_evidence = _MODEL_EXTRACTOR.extract_signals(
                text, startup_name=entry.get("startup_name", "")
            )
            for signal_name, model_level in model_vals.items():
                if model_level:  # we let the model win whenever it produces a result
                    entry[signal_name] = model_level
                    if signal_name in model_evidence:
                        entry["signal_evidence"][signal_name] = model_evidence[signal_name]

        return entry

    def _extract_business_model(self, text: str, metadata: dict) -> str:
        """Detect business model using hardened rules."""
        # we check YC metadata tags first because they often carry explicit business model hints
        tags = metadata.get("tags", [])
        if tags:
            tag_text = " ".join(str(t) for t in tags)
            model = self._classify_business_model(tag_text + " " + text[:1000])
            if model:
                return model

        return self._classify_business_model(text)

    def _extract_target_user(self, text: str, metadata: dict) -> str:
        """Extract target user from text or metadata."""
        patterns = [
            r"(?:for|serving|targeting)\s+([^\.]{5,80})",
            r"(?:help(?:s|ing)?|enable(?:s)?|empower(?:s)?)\s+([^\.]{5,80})",
        ]

        for pattern in patterns:
            match = re.search(pattern, text[:500], re.IGNORECASE)
            if match:
                target = match.group(1).strip()
                return truncate(clean_text(target), 150)

        return ""

    def _extract_value_proposition(self, text: str, metadata: dict) -> str:
        """Extract value proposition -- prioritize first lines of content."""
        lines = text.split("\n")
        first_line = lines[0].strip() if lines else ""

        if first_line and 10 < len(first_line) < 300:
            return first_line

        match = re.search(r"([A-Z][^\.]{20,200}\.)", text[:1000])
        if match:
            return truncate(clean_text(match.group(1)), 300)

        return ""

    def _extract_failure_reasons_failory(self, text: str) -> list[str]:
        """
        Extract failure reasons from Failory case studies.
        Enhanced Phase 2 with fallback parsing and structured extraction.
        """
        reasons = []
        text_lower = text.lower()

        # we scan keyword categories with a minimum hit threshold to avoid false positives
        for category, keywords in FAILURE_KEYWORDS.items():
            # we count both exact matches and contextual presence to score each category
            score = 0
            matched_kws = []
            for kw in keywords:
                if kw.lower() in text_lower:
                    score += 1
                    matched_kws.append(kw)

            if score >= 2:  # we require at least 2 keyword hits before adding the category
                phrases = self._extract_matching_phrases(text, matched_kws[:3], context_window=120)
                if phrases:
                    reason = f"{self._format_failure_category(category)}: {truncate(clean_text(phrases[0]), 200)}"
                    reasons.append(reason)
                else:
                    reasons.append(self._format_failure_category(category))

        # we also run explicit cause-language patterns to catch failures not covered by keyword scoring
        explicit_patterns = [
            r"(?:failed|shut down|closed|died)\s+because\s+([^\.]{10,150})",
            r"(?:reason|cause)\s+(?:for|of)\s+(?:the\s+)?(?:failure|shutdown|closing)[:\s]+([^\.]{10,150})",
            r"(?:main\s+)?(?:mistake|lesson|error)\s+(?:was|were)[:\s]+([^\.]{10,150})",
            r"what\s+went\s+wrong[:\s]*([^\.]{10,150})",
            r"why\s+(?:we|they|it)\s+failed[:\s]*([^\.]{10,150})",
            r"(?:the\s+)?(?:biggest|main|key)\s+(?:reason|factor|cause)[:\s]*([^\.]{10,150})",
            r"(?:ultimately|eventually)\s+(?:failed|collapsed|shut down)\s+(?:because|due to)\s+([^\.]{10,150})",
        ]

        for pattern in explicit_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match_text in matches:
                cleaned = truncate(clean_text(match_text), 200)
                if cleaned and len(cleaned) > 15:
                    # we deduplicate here to avoid surfacing the same reason twice
                    if not any(cleaned.lower() in r.lower() or r.lower() in cleaned.lower()
                              for r in reasons):
                        reasons.append(cleaned)

        # we add a "lesson" section fallback because Failory articles often close with explicit takeaways
        lesson_patterns = [
            r"(?:key\s+)?lessons?\s+(?:learned|from)[:\s]*([^\.]{10,200})",
            r"takeaway[s]?[:\s]*([^\.]{10,200})",
            r"what\s+(?:we|they|founders?)\s+learned[:\s]*([^\.]{10,200})",
        ]

        for pattern in lesson_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match_text in matches[:2]:
                cleaned = truncate(clean_text(match_text), 200)
                if cleaned and len(cleaned) > 15:
                    if not any(cleaned.lower() in r.lower() for r in reasons):
                        reasons.append(f"Lesson: {cleaned}")

        return reasons[:8]

    def _extract_failure_reasons_generic(self, text: str) -> list[str]:
        """Extract failure reasons from non-Failory sources."""
        reasons = []
        text_lower = text.lower()

        # we guard extraction behind explicit failure language so non-failure entries stay clean
        failure_signals = ["failed", "shut down", "closed", "didn't work",
                          "ran out", "went wrong", "mistake", "collapsed"]

        if not any(s in text_lower for s in failure_signals):
            return []

        for category, keywords in FAILURE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score >= 3:
                reasons.append(self._format_failure_category(category))

        return reasons[:5]

    def _format_failure_category(self, category: str) -> str:
        """Format a failure category key into readable text."""
        return category.replace("_", " ").title()

    def _extract_success_drivers(self, text: str, metadata: dict) -> list[str]:
        """Extract success drivers from text and metadata."""
        drivers = []

        # we check YC metadata for quick success signals before scanning the full text
        if metadata.get("top_company"):
            drivers.append("Y Combinator top company")
        if metadata.get("status") == "Active":
            drivers.append("Active and operational")
        team_size = metadata.get("team_size", 0)
        if isinstance(team_size, int) and team_size > 50:
            drivers.append(f"Scaled to {team_size}+ team members")

        # we also scan the raw text for success language that isn't captured in metadata
        phrases = self._extract_matching_phrases(text, SUCCESS_KEYWORDS, context_window=100)
        for phrase in phrases[:5]:
            cleaned = truncate(clean_text(phrase), 200)
            if cleaned and len(cleaned) > 15:
                drivers.append(cleaned)

        return drivers[:6]

    def _extract_pain_points(self, text: str) -> list[str]:
        """Extract pain points mentioned in the content."""
        phrases = self._extract_matching_phrases(text, PAIN_KEYWORDS, context_window=100)
        clean_pains = []
        for phrase in phrases:
            cleaned = truncate(clean_text(phrase), 200)
            if cleaned and len(cleaned) > 15:
                clean_pains.append(cleaned)
        return clean_pains[:5]

    def _extract_funding_stage(self, text: str) -> str:
        """Extract the funding stage at the time of failure/event."""
        text_lower = text.lower()

        for stage, keywords in FUNDING_STAGE_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return stage.replace("_", " ").title()

        return ""

    def _extract_market_context(self, text: str) -> str:
        """
        Extract brief market context from Failory articles.
        Looks for descriptions of the market/space the startup operated in.
        """
        market_patterns = [
            r"(?:the\s+)?(?:market|industry|space|sector)\s+(?:was|is|for)\s+([^\.]{15,200})",
            r"(?:operating|operated|working)\s+in\s+(?:the\s+)?([^\.]{10,150})\s+(?:market|space|industry|sector)",
            r"(?:valued|worth|estimated)\s+at\s+\$?([^\.]{5,80})",
            r"(?:the\s+)?(\$[\d\.]+\s*(?:billion|million|trillion)\s+(?:market|industry))",
            r"(?:growing|grew|growth)\s+(?:at|by)\s+([^\.]{5,80})",
        ]

        for pattern in market_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return truncate(clean_text(match.group(1)), 200)

        return ""

    def _extract_differentiation(self, text: str) -> str:
        """Extract differentiation from insight content."""
        diff_keywords = [
            "unlike", "different from", "unique", "innovative",
            "first to", "disrupted", "revolutionized", "novel approach",
            "differentiated by", "competitive advantage",
        ]
        phrases = self._extract_matching_phrases(text, diff_keywords, context_window=120)
        if phrases:
            return truncate(clean_text(phrases[0]), 300)
        return ""
