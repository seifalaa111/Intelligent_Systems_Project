"""
MIDAN Data Pipeline -- Schema Validator (Phase 2)
Validates, cleans, and NORMALIZES structured entries.
Enforces: no hallucination, no generic filler, no macro data, consistent labels.
"""

import re
from utils.logger import setup_logger
from config.settings import (
    LOG_FILE, BUSINESS_MODEL_CANONICAL, INDUSTRY_CANONICAL,
    SWITCHING_COST_CANONICAL,
)

logger = setup_logger("schema_validator", LOG_FILE)

# Valid source types (including multi-source composites)
VALID_SOURCE_TYPES = {"website", "insight", "directory", "community", "review"}

# Valid L3 signal values
VALID_RETENTION = {"high", "medium", "low", ""}
VALID_ONBOARDING = {"low", "medium", "high", ""}
VALID_MONETIZATION = {"strong", "moderate", "weak", ""}
VALID_COMPETITION = {"high", "moderate", "low", ""}

# Generic filler patterns
GENERIC_FILLER_PATTERNS = [
    r"^n/?a$",
    r"^none$",
    r"^unknown$",
    r"^not available$",
    r"^not specified$",
    r"^to be determined$",
    r"^tbd$",
    r"^lorem ipsum",
    r"^example",
    r"^placeholder",
    r"^test\s",
    r"^generic\s",
    r"^this is a\s",
    r"^the company\s+(is|was|does)\s",
]

# Macroeconomic terms that must NOT appear
MACRO_BLACKLIST = [
    "gdp", "gross domestic product", "inflation rate", "cpi",
    "consumer price index", "federal reserve", "interest rate",
    "money supply", "fiscal policy", "monetary policy",
    "unemployment rate", "trade deficit", "national debt",
    "quantitative easing", "macroeconomic",
]


def normalize_entry(entry: dict) -> dict:
    """
    Normalize all labels to canonical forms.
    This is the SINGLE place where label consistency is enforced.
    """
    # ── BUSINESS MODEL NORMALIZATION ──
    bm = entry.get("business_model", "").strip()
    if bm:
        bm_lower = bm.lower()
        # Check canonical mapping
        canonical = BUSINESS_MODEL_CANONICAL.get(bm_lower)
        if canonical:
            entry["business_model"] = canonical
        else:
            # Try to match partial keys
            for key, val in BUSINESS_MODEL_CANONICAL.items():
                if key in bm_lower or bm_lower in key:
                    entry["business_model"] = val
                    break
            else:
                # Already in a good format, just title-case it
                entry["business_model"] = bm.title() if bm.isupper() else bm

    # ── INDUSTRY NORMALIZATION ──
    for ind_field in ["primary_industry", "secondary_industry"]:
        ind = entry.get(ind_field, "").strip()
        if ind:
            ind_lower = ind.lower()
            canonical = INDUSTRY_CANONICAL.get(ind_lower)
            if canonical:
                entry[ind_field] = canonical
            else:
                for key, val in INDUSTRY_CANONICAL.items():
                    if key in ind_lower or ind_lower in key:
                        entry[ind_field] = val
                        break

    # ── TARGET SEGMENT NORMALIZATION ──
    seg = entry.get("target_segment", "").strip().upper()
    if seg and seg not in ("B2B", "B2C", "B2B + B2C"):
        entry["target_segment"] = ""
    elif seg:
        entry["target_segment"] = seg

    # ── SWITCHING COST NORMALIZATION ──
    sc = entry.get("switching_cost", "").strip().lower()
    entry["switching_cost"] = sc if sc in SWITCHING_COST_CANONICAL else ""

    # ── L3 SIGNALS NORMALIZATION & EVIDENCE CHECK ──
    def _normalize_signal(field_name, valid_set):
        val = entry.get(field_name, "").strip().lower()
        if val not in valid_set:
            entry[field_name] = ""
            if "signal_evidence" in entry and field_name in entry["signal_evidence"]:
                entry["signal_evidence"].pop(field_name)
        else:
            entry[field_name] = val
            # Enforce evidence -> if no evidence, clear the signal
            if "signal_evidence" not in entry or not entry["signal_evidence"].get(field_name):
                entry[field_name] = ""

    _normalize_signal("retention_proxy", VALID_RETENTION)
    _normalize_signal("onboarding_friction", VALID_ONBOARDING)
    _normalize_signal("monetization_strength", VALID_MONETIZATION)
    _normalize_signal("competition_intensity", VALID_COMPETITION)
    _normalize_signal("competition_density", VALID_COMPETITION)

    return entry


def validate_entry(entry: dict) -> tuple[bool, list[str]]:
    """
    Validate a single structured entry against the schema.
    Returns: (is_valid, list_of_issues)
    """
    issues = []

    # ── REQUIRED FIELD: startup_name ──
    if not entry.get("startup_name") or not entry["startup_name"].strip():
        issues.append("CRITICAL: Missing startup_name")
        return False, issues

    # ── VALID SOURCE TYPE ──
    source_type = entry.get("source_type", "")
    if source_type and not source_type.startswith("multi:"):
        if source_type not in VALID_SOURCE_TYPES:
            issues.append(f"Invalid source_type: '{source_type}'")

    # ── TYPE CHECKS ──
    string_fields = ["startup_name", "primary_industry", "secondary_industry", "business_model", "target_user",
                     "target_segment", "value_proposition", "differentiation",
                     "switching_cost", "competition_density", "source_type",
                     "source_url", "retention_proxy", "onboarding_friction",
                     "monetization_strength", "competition_intensity",
                     "funding_stage", "market_context"]
    list_fields = ["pain_points", "adoption_barriers", "user_complaints",
                   "success_drivers", "failure_reasons"]

    for field in string_fields:
        if field in entry and not isinstance(entry[field], str):
            issues.append(f"Type error: '{field}' must be string, got {type(entry[field]).__name__}")

    for field in list_fields:
        if field in entry and not isinstance(entry[field], list):
            issues.append(f"Type error: '{field}' must be list, got {type(entry[field]).__name__}")

    # ── GENERIC FILLER CHECK ──
    for field in string_fields:
        value = entry.get(field, "")
        if value and _is_generic_filler(value):
            issues.append(f"Generic filler detected in '{field}': '{value[:50]}'")
            entry[field] = ""

    for field in list_fields:
        values = entry.get(field, [])
        cleaned = [v for v in values if not _is_generic_filler(v)]
        if len(cleaned) < len(values):
            issues.append(f"Removed {len(values) - len(cleaned)} generic fillers from '{field}'")
        entry[field] = cleaned

    # ── MACRO DATA CHECK ──
    macro_violations = _check_macro_data(entry)
    if macro_violations:
        issues.extend(macro_violations)

    # ── CONTENT QUALITY ──
    has_any_content = False
    for field in string_fields[1:]:
        if entry.get(field, "").strip():
            has_any_content = True
            break
    for field in list_fields:
        if entry.get(field, []):
            has_any_content = True
            break

    if not has_any_content:
        issues.append("WARNING: Entry has no substantive content beyond startup_name")

    is_valid = not any("CRITICAL" in issue for issue in issues)
    return is_valid, issues


def validate_batch(entries: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Validate a batch of entries. Returns (valid, invalid, stats)."""
    valid = []
    invalid = []
    all_issues = []

    for entry in entries:
        is_valid, issues = validate_entry(entry)
        if is_valid:
            valid.append(entry)
        else:
            invalid.append({"entry": entry, "issues": issues})
        all_issues.extend(issues)

    stats = {
        "total": len(entries),
        "valid": len(valid),
        "invalid": len(invalid),
        "total_issues": len(all_issues),
        "issue_summary": _summarize_issues(all_issues),
    }

    logger.info(f"Validation: {stats['valid']}/{stats['total']} valid, "
                f"{stats['total_issues']} issues found")

    return valid, invalid, stats


def clean_entry(entry: dict) -> dict:
    """
    Clean, normalize, and ensure schema compliance for a structured entry.
    This is the SINGLE entry point for all post-extraction cleaning.
    """
    cleaned = {
        "startup_name": str(entry.get("startup_name", "")).strip(),
        "primary_industry": str(entry.get("primary_industry", "")).strip(),
        "secondary_industry": str(entry.get("secondary_industry", "")).strip(),
        "business_model": str(entry.get("business_model", "")).strip(),
        "target_user": str(entry.get("target_user", "")).strip(),
        "target_segment": str(entry.get("target_segment", "")).strip(),
        "value_proposition": str(entry.get("value_proposition", "")).strip(),
        "pain_points": _clean_list(entry.get("pain_points", [])),
        "adoption_barriers": _clean_list(entry.get("adoption_barriers", [])),
        "user_complaints": _clean_list(entry.get("user_complaints", [])),
        "success_drivers": _clean_list(entry.get("success_drivers", [])),
        "failure_reasons": _clean_list(entry.get("failure_reasons", [])),
        "differentiation": str(entry.get("differentiation", "")).strip(),
        "switching_cost": str(entry.get("switching_cost", "")).strip(),
        "competition_density": str(entry.get("competition_density", "")).strip(),
        "signal_evidence": entry.get("signal_evidence", {}),
        "confidence_score": float(entry.get("confidence_score", 0.0)),
        # Phase 2 L3 signals
        "retention_proxy": str(entry.get("retention_proxy", "")).strip(),
        "onboarding_friction": str(entry.get("onboarding_friction", "")).strip(),
        "monetization_strength": str(entry.get("monetization_strength", "")).strip(),
        "competition_intensity": str(entry.get("competition_intensity", "")).strip(),
        "funding_stage": str(entry.get("funding_stage", "")).strip(),
        "market_context": str(entry.get("market_context", "")).strip(),
        "source_type": str(entry.get("source_type", "")).strip(),
        "source_url": str(entry.get("source_url", "")).strip(),
    }

    # Apply normalization AFTER cleaning
    cleaned = normalize_entry(cleaned)

    return cleaned


def _is_generic_filler(text: str) -> bool:
    if not text:
        return False
    text_lower = text.strip().lower()
    for pattern in GENERIC_FILLER_PATTERNS:
        if re.match(pattern, text_lower):
            return True
    return False


def _check_macro_data(entry: dict) -> list[str]:
    violations = []
    all_text = ""
    for field, value in entry.items():
        if isinstance(value, str):
            all_text += " " + value
        elif isinstance(value, list):
            all_text += " " + " ".join(str(v) for v in value)
    all_text_lower = all_text.lower()
    for term in MACRO_BLACKLIST:
        if term in all_text_lower:
            violations.append(f"MACRO DATA VIOLATION: Found '{term}' in entry for "
                            f"'{entry.get('startup_name', 'unknown')}'")
    return violations


def _clean_list(items: list) -> list[str]:
    if not isinstance(items, list):
        return []
    seen = set()
    cleaned = []
    for item in items:
        s = str(item).strip()
        if s and s.lower() not in seen and not _is_generic_filler(s):
            seen.add(s.lower())
            cleaned.append(s)
    return cleaned


def _summarize_issues(issues: list[str]) -> dict:
    summary = {}
    for issue in issues:
        category = issue.split(":")[0].strip()
        summary[category] = summary.get(category, 0) + 1
    return summary
