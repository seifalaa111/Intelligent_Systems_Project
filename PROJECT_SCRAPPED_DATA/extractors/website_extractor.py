"""
MIDAN Data Pipeline -- Website Extractor (Phase 2)
Rule-based extraction of business model, value proposition, target user
from startup website content. Uses hardened classification engine.
"""

import re
from extractors.base_extractor import BaseExtractor
from utils.text_cleaner import clean_text, truncate
from config.settings import PAIN_KEYWORDS

# Model-assisted signal extractor (replaces rule-based _classify_level for L3)
try:
    from extractors.model_signal_extractor import get_model_signal_extractor
    _MODEL_EXTRACTOR = get_model_signal_extractor()
except Exception:
    _MODEL_EXTRACTOR = None


class WebsiteExtractor(BaseExtractor):
    """
    Extracts structured intelligence from startup website content.

    Primary fields: business_model, industry, target_user, value_proposition
    L3 signals: retention_proxy, onboarding_friction, monetization_strength,
                competition_intensity, switching_cost
    """

    def __init__(self):
        super().__init__("website")

    def extract(self, raw_entry: dict) -> dict:
        """Extract structured fields from website raw content."""
        entry = self._empty_entry()
        entry["startup_name"] = raw_entry.get("startup_name", "")
        entry["source_url"] = raw_entry.get("source_url", "")
        entry["source_type"] = "website"

        raw_content = raw_entry.get("raw_content", "")
        metadata = raw_entry.get("metadata", {})
        meta_tags = metadata.get("meta_tags", {})
        headings = metadata.get("headings", [])

        if not raw_content:
            return entry

        text = clean_text(raw_content)

        # Build enriched search text (meta + content)
        search_text = self._build_search_text(text, meta_tags)

        # ── CORE FIELDS ──
        entry["primary_industry"], entry["secondary_industry"] = self._classify_industry(search_text)
        entry["business_model"] = self._classify_business_model(search_text)
        entry["target_user"] = self._extract_target_user(text, headings)
        entry["target_segment"] = self._classify_target_segment(search_text)
        entry["value_proposition"] = self._extract_value_proposition(text, meta_tags, headings)
        entry["differentiation"] = self._extract_differentiation(text, headings)

        # ── PAIN POINTS (problems the startup claims to solve) ──
        entry["pain_points"] = self._extract_pain_points(text)
        entry["adoption_barriers"] = self._extract_adoption_barriers(text)

        # ── L3 SIGNALS ──
        # Step 1: rule-based baseline (original approach)
        def add_signal(field_name, result_tuple):
            level, evidence = result_tuple
            entry[field_name] = level
            if level and evidence:
                entry["signal_evidence"][field_name] = evidence

        add_signal("switching_cost", self._extract_switching_cost(text))
        add_signal("retention_proxy", self._extract_retention_proxy(text))
        add_signal("onboarding_friction", self._extract_onboarding_friction(text))
        add_signal("monetization_strength", self._extract_monetization_strength(text))
        add_signal("competition_intensity", self._extract_competition_intensity(text))

        # Step 2: model-assisted override — model results take priority over rule-based.
        # BOUNDARY: ModelSignalExtractor reads ONLY raw text. It never sees
        # decision_analysis, system_patterns, or confidence_score.
        if _MODEL_EXTRACTOR is not None:
            model_vals, model_evidence = _MODEL_EXTRACTOR.extract_signals(
                text, startup_name=entry.get("startup_name", "")
            )
            for signal_name, model_level in model_vals.items():
                if model_level and signal_name != "switching_cost":
                    entry[signal_name] = model_level
                    if signal_name in model_evidence:
                        entry["signal_evidence"][signal_name] = model_evidence[signal_name]

        return entry

    def _build_search_text(self, text: str, meta_tags: dict) -> str:
        """Combine meta tags and content for richer classification signal."""
        parts = []
        if meta_tags.get("description"):
            parts.append(meta_tags["description"])
        if meta_tags.get("keywords"):
            parts.append(meta_tags["keywords"])
        if meta_tags.get("og_description"):
            parts.append(meta_tags["og_description"])
        if meta_tags.get("title"):
            parts.append(meta_tags["title"])
        parts.append(text[:3000])  # cap to first 3000 chars for classification
        return " ".join(parts)

    def _extract_target_user(self, text: str, headings: list[str]) -> str:
        """Extract target user/audience description."""
        patterns = [
            r"(?:built|designed|made|created)\s+for\s+([^\.]{10,80})",
            r"(?:helping|help|enables?|empowers?)\s+([^\.]{10,80})",
            r"for\s+(teams?|businesses?|developers?|startups?|enterprises?|creators?|"
            r"marketers?|designers?|professionals?|students?|teachers?|"
            r"small businesses?|freelancers?|agencies?|companies?)[^\.]{0,60}",
            r"(?:platform|tool|solution|software)\s+for\s+([^\.]{10,80})",
        ]

        for pattern in patterns:
            match = re.search(pattern, text[:2000], re.IGNORECASE)
            if match:
                target = match.group(1).strip()
                target = re.sub(r"\s+", " ", target)
                return truncate(clean_text(target), 150)

        for heading in headings[:5]:
            heading_lower = heading.lower()
            if any(word in heading_lower for word in ["for teams", "for businesses",
                                                       "for developers", "for creators",
                                                       "for everyone"]):
                return heading

        return ""

    def _extract_value_proposition(self, text: str, meta_tags: dict,
                                    headings: list[str]) -> str:
        """Extract the core value proposition."""
        # Priority 1: Meta description
        if meta_tags.get("description"):
            desc = meta_tags["description"].strip()
            if 20 < len(desc) < 300:
                return desc

        # Priority 2: OG description
        if meta_tags.get("og_description"):
            og_desc = meta_tags["og_description"].strip()
            if 20 < len(og_desc) < 300:
                return og_desc

        # Priority 3: First meaningful heading
        if headings:
            for h in headings[:3]:
                if len(h) > 15 and len(h) < 200:
                    return h

        # Priority 4: First sentence patterns
        vp_patterns = [
            r"([A-Z][^\.]{20,150}(?:better|faster|easier|simpler|smarter)[^\.]{0,50}\.)",
            r"([A-Z][^\.]{20,150}(?:help|enable|empower|transform|automate)[^\.]{0,50}\.)",
            r"^([A-Z][^\.]{30,200}\.)",
        ]

        for pattern in vp_patterns:
            match = re.search(pattern, text[:2000])
            if match:
                return truncate(clean_text(match.group(1)), 300)

        return ""

    def _extract_differentiation(self, text: str, headings: list[str]) -> str:
        """Extract what makes the startup different."""
        diff_keywords = [
            "unlike", "different from", "only platform", "first to",
            "unique", "patent", "proprietary", "no other", "the only",
            "competitive advantage", "what sets us apart", "why choose",
        ]

        phrases = self._extract_matching_phrases(text, diff_keywords, context_window=120)
        if phrases:
            return truncate(clean_text(phrases[0]), 300)

        for heading in headings:
            heading_lower = heading.lower()
            if any(w in heading_lower for w in ["why", "different", "unique", "better"]):
                return heading

        return ""

    def _extract_pain_points(self, text: str) -> list[str]:
        """Extract pain points the startup claims to solve."""
        pain_phrases = self._extract_matching_phrases(text[:3000], PAIN_KEYWORDS, context_window=100)
        clean_pains = []
        for phrase in pain_phrases:
            cleaned = clean_text(phrase)
            if cleaned and len(cleaned) > 20:
                clean_pains.append(truncate(cleaned, 200))
        return clean_pains[:5]

    def _extract_adoption_barriers(self, text: str) -> list[str]:
        """Infer adoption barriers from website signals."""
        barriers = []
        text_lower = text.lower()

        if any(w in text_lower for w in ["enterprise pricing", "contact sales",
                                          "request a demo", "custom pricing"]):
            barriers.append("Complex or opaque pricing -- requires sales contact")

        if any(w in text_lower for w in ["integration required", "api setup",
                                          "technical implementation", "developer needed"]):
            barriers.append("Technical setup required -- developer dependency")

        if any(w in text_lower for w in ["migration", "import your data",
                                          "switch from", "transfer"]):
            barriers.append("Data migration required from existing tools")

        if any(w in text_lower for w in ["documentation", "getting started guide",
                                          "tutorials", "training", "onboarding"]):
            barriers.append("Learning curve -- onboarding/training needed")

        return barriers[:4]

    def _extract_switching_cost(self, text: str) -> tuple[str, list[str]]:
        """Infer switching cost level from text signals."""
        text_lower = text.lower()

        high_signals = ["data migration", "integration", "enterprise contract",
                       "annual plan", "custom setup", "sso", "compliance",
                       "api integration", "dedicated support"]
        medium_signals = ["import", "export", "api", "team plan",
                         "workspace", "connect your"]
        low_signals = ["free plan", "no credit card", "instant setup",
                      "no commitment", "cancel anytime", "free trial",
                      "get started free"]

        high_score = sum(1 for s in high_signals if s in text_lower)
        medium_score = sum(1 for s in medium_signals if s in text_lower)
        low_score = sum(1 for s in low_signals if s in text_lower)

        level = ""
        evidence_kws = []
        if high_score >= 2:
            level = "high"
            evidence_kws = high_signals
        elif medium_score >= 2 or high_score >= 1:
            level = "medium"
            evidence_kws = medium_signals + high_signals
        elif low_score >= 2:
            level = "low"
            evidence_kws = low_signals

        if level:
            evidence = self._extract_matching_phrases(text, evidence_kws, context_window=100)
            return level, evidence[:2]
        return "", []
