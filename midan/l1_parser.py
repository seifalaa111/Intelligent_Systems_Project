"""
midan.l1_parser — confidence-scored idea feature extraction.

Returns a structured envelope `{values, confidence, source, is_sufficient,
consistency, ...}` where invalid/low-confidence fields become the literal
UNKNOWN sentinel rather than silently defaulted. Cross-field consistency
enforced. Heuristic fallback paths emit explicit confidence per field.
"""
from midan.core import *  # noqa: F401,F403


# ── extracted from api.py ─────────────────────────────────────────────



# ═══════════════════════════════════════════════════════════════
# AGENT A1 — NLP Parser (sector + country from free text)
# ═══════════════════════════════════════════════════════════════

def agent_a1_parse(idea_text: str):
    t = idea_text.lower()
    sector_scores = _score_sector_candidates(t)
    sector, best_score = max(
        sector_scores.items(),
        key=lambda item: (item[1], SECTOR_TIEBREAKER.get(item[0], 0))
    )
    sector_found = best_score > 0
    if not sector_found:
        sector = 'fintech'
    country, country_found = None, False
    for code, kws in COUNTRY_KEYWORDS.items():
        if any(k in t for k in kws):
            country, country_found = code, True
            break
    if not country_found:
        country = 'EG'
    return sector, country, sector_found, country_found

# ═══════════════════════════════════════════════════════════════
# LAYER 1 — IDEA FEATURE EXTRACTION
# Converts raw idea text → structured 8-field schema.
# This is what makes different ideas produce different outputs.
# ═══════════════════════════════════════════════════════════════

BUSINESS_MODELS    = ['subscription', 'marketplace', 'saas', 'commission', 'service', 'hardware', 'other']
TARGET_SEGMENTS    = ['b2b', 'b2c', 'b2g', 'mixed']
MONETIZATION_TYPES = ['subscription', 'commission', 'one-time', 'freemium', 'ad-based', 'other']
STAGES             = ['idea', 'validation', 'mvp', 'growth']
INTENSITY_LEVELS   = ['low', 'medium', 'high']

# Required L1 fields — the pipeline halts if any of these are UNKNOWN after extraction.
# Unrequired fields can be UNKNOWN without blocking analysis (they degrade gracefully downstream).
L1_REQUIRED_FIELDS = ['business_model', 'target_segment', 'stage']
# L1_MIN_FIELD_CONFIDENCE, L1_MIN_AGGREGATE_CONFIDENCE and UNKNOWN_VALUE are
# owned by midan.config and reach this module via `from midan.core import *`.

_L1_LOG = __import__('logging').getLogger("midan.l1")


def _l1_unknown_field(name: str, reason: str = "low_confidence") -> dict:
    """Build an UNKNOWN field record. Numerics get None, enums get the UNKNOWN sentinel."""
    is_numeric = name in {'differentiation_score', 'market_readiness'}
    return {
        "value": None if is_numeric else UNKNOWN_VALUE,
        "confidence": 0.0,
        "source": "unknown",
        "reason": reason,
    }


def _l1_field(value, confidence: float, source: str) -> dict:
    return {"value": value, "confidence": float(confidence), "source": source}


def _coerce_enum(raw, allowed: list, name: str, llm_conf: float) -> dict:
    """Validate an enum field. Invalid value or sub-threshold confidence → UNKNOWN."""
    if raw not in allowed:
        return _l1_unknown_field(name, "invalid_enum")
    if llm_conf < L1_MIN_FIELD_CONFIDENCE:
        return _l1_unknown_field(name, "low_llm_confidence")
    return _l1_field(raw, llm_conf, "llm")


def _coerce_score(raw, name: str, llm_conf: float) -> dict:
    """Validate a 1-5 score. Out-of-range or sub-threshold → UNKNOWN."""
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return _l1_unknown_field(name, "non_numeric")
    if not (1 <= v <= 5):
        return _l1_unknown_field(name, "out_of_range")
    if llm_conf < L1_MIN_FIELD_CONFIDENCE:
        return _l1_unknown_field(name, "low_llm_confidence")
    return _l1_field(v, llm_conf, "llm")


# Neutral defaults used ONLY for non-required UNKNOWN fields when feeding the
# values into L2/L3/L4. The L1 envelope's `values` dict keeps the UNKNOWN
# marker — these defaults exist solely so downstream arithmetic does not
# operate on None/UNKNOWN. Required fields with UNKNOWN halt the pipeline
# before this default is ever consumed.
_L1_NEUTRAL_DEFAULTS = {
    "monetization":          "other",
    "differentiation_score": 3,
    "competitive_intensity": "medium",
    "regulatory_risk":       "medium",
    "market_readiness":      3,
}


def _result_from_fields(fields: dict, sector: str, idea_text: str) -> dict:
    """Assemble the L1 result envelope from a dict of per-field records."""
    values     = {k: v["value"]      for k, v in fields.items()}
    confidence = {k: v["confidence"] for k, v in fields.items()}
    source     = {k: v["source"]     for k, v in fields.items()}

    unknown_required = [
        f for f in L1_REQUIRED_FIELDS
        if values.get(f) in (UNKNOWN_VALUE, None)
    ]
    required_confs = [confidence[f] for f in L1_REQUIRED_FIELDS if f in confidence]
    aggregate_conf = (sum(required_confs) / len(required_confs)) if required_confs else 0.0

    consistency = _validate_l1_consistency(values, sector, idea_text)

    is_sufficient = (
        not unknown_required
        and aggregate_conf >= L1_MIN_AGGREGATE_CONFIDENCE
        and consistency["ok"]
    )

    # Runtime view: required UNKNOWN are NOT defaulted (they halt the gate).
    # Non-required UNKNOWN get a documented neutral default so L2/L3/L4
    # arithmetic does not see None — but `values` and `source` still report
    # UNKNOWN so the API response remains honest.
    runtime_values: dict = {}
    for k, v in values.items():
        if v in (UNKNOWN_VALUE, None) and k not in L1_REQUIRED_FIELDS:
            runtime_values[k] = _L1_NEUTRAL_DEFAULTS.get(k, v)
        else:
            runtime_values[k] = v

    return {
        "values":              values,
        "runtime_values":      runtime_values,
        "confidence":          confidence,
        "source":              source,
        "unknown_required":    unknown_required,
        "aggregate_confidence": round(aggregate_conf, 3),
        "consistency":         consistency,
        "is_sufficient":       is_sufficient,
        "min_field_threshold": L1_MIN_FIELD_CONFIDENCE,
        "min_aggregate_threshold": L1_MIN_AGGREGATE_CONFIDENCE,
    }


def extract_idea_features(idea_text: str, sector: str) -> dict:
    """
    Layer 1 — Structured extraction with first-class confidence.

    Returns an envelope:
        {
          "values":        {<field>: value | "UNKNOWN" | None},
          "confidence":    {<field>: 0.0–1.0},
          "source":        {<field>: "llm" | "heuristic" | "unknown"},
          "unknown_required": [<required fields below threshold>],
          "aggregate_confidence": float,
          "consistency":   {"ok": bool, "violations": [...]},
          "is_sufficient": bool,
        }

    Low-confidence or invalid fields are NOT silently coerced to defaults —
    they are marked UNKNOWN. The pipeline halts upstream when is_sufficient=False.
    """
    if not idea_text or len(idea_text.strip()) < 8:
        # Too little text for any meaningful extraction. Mark every field UNKNOWN
        # so the gate halts the pipeline.
        fields = {name: _l1_unknown_field(name, "input_too_short") for name in (
            'business_model', 'target_segment', 'monetization', 'stage',
            'differentiation_score', 'competitive_intensity',
            'regulatory_risk', 'market_readiness',
        )}
        return _result_from_fields(fields, sector, idea_text)

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if GROQ_CLIENT and groq_key and groq_key != "dummy":
        try:
            prompt = dedent(f"""
                Extract structured startup features from this idea description.
                Return ONLY valid JSON matching the exact schema. No explanation.

                Idea: "{idea_text}"
                Sector hint: {sector}

                For EVERY field, also emit a confidence score 0.0–1.0 reflecting how
                certain you are based on the idea text. Use < 0.5 when the field is
                not stated or implied — do NOT guess.

                Required JSON:
                {{
                  "business_model": "subscription|marketplace|saas|commission|service|hardware|other",
                  "business_model_confidence": 0.0,
                  "target_segment": "b2b|b2c|b2g|mixed",
                  "target_segment_confidence": 0.0,
                  "monetization": "subscription|commission|one-time|freemium|ad-based|other",
                  "monetization_confidence": 0.0,
                  "stage": "idea|validation|mvp|growth",
                  "stage_confidence": 0.0,
                  "differentiation_score": 3,
                  "differentiation_score_confidence": 0.0,
                  "competitive_intensity": "low|medium|high",
                  "competitive_intensity_confidence": 0.0,
                  "regulatory_risk": "low|medium|high",
                  "regulatory_risk_confidence": 0.0,
                  "market_readiness": 3,
                  "market_readiness_confidence": 0.0
                }}

                Rules:
                - Use ONLY the exact enum values listed above.
                - Confidence rubric: 0.9+ explicit in text, 0.7 strongly implied,
                  0.5 plausibly inferred, < 0.5 not stated (mark unsure).
                - Different ideas MUST produce different values — do NOT default to all 3s.
                - If the idea text does not provide a signal for a field, give a low
                  confidence rather than guessing — UNKNOWN is preferable to a wrong default.
            """).strip()

            resp = GROQ_CLIENT.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=GROQ_MODEL,
                temperature=0.0,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content.strip())

            def _conf(name: str, default: float = 0.0) -> float:
                try:
                    c = float(data.get(f"{name}_confidence", default))
                except (TypeError, ValueError):
                    return 0.0
                return max(0.0, min(1.0, c))

            fields = {
                'business_model':        _coerce_enum(data.get('business_model'),        BUSINESS_MODELS,    'business_model',        _conf('business_model')),
                'target_segment':        _coerce_enum(data.get('target_segment'),        TARGET_SEGMENTS,    'target_segment',        _conf('target_segment')),
                'monetization':          _coerce_enum(data.get('monetization'),          MONETIZATION_TYPES, 'monetization',          _conf('monetization')),
                'stage':                 _coerce_enum(data.get('stage'),                 STAGES,             'stage',                 _conf('stage')),
                'competitive_intensity': _coerce_enum(data.get('competitive_intensity'), INTENSITY_LEVELS,   'competitive_intensity', _conf('competitive_intensity')),
                'regulatory_risk':       _coerce_enum(data.get('regulatory_risk'),       INTENSITY_LEVELS,   'regulatory_risk',       _conf('regulatory_risk')),
                'differentiation_score': _coerce_score(data.get('differentiation_score'), 'differentiation_score', _conf('differentiation_score')),
                'market_readiness':      _coerce_score(data.get('market_readiness'),      'market_readiness',      _conf('market_readiness')),
            }
            result = _result_from_fields(fields, sector, idea_text)

            # Low-confidence required fields → degrade with heuristic ONLY when
            # the heuristic has stronger keyword evidence. Otherwise leave UNKNOWN.
            if result["unknown_required"]:
                result = _backfill_with_heuristic(result, idea_text, sector)
            return result
        except Exception as llm_err:
            _L1_LOG.warning(
                f"[L1] extract_idea_features LLM failed "
                f"({type(llm_err).__name__}: {llm_err!r}) — falling back to heuristic"
            )

    return _heuristic_idea_features(idea_text, sector)


def _heuristic_field(name: str, value, evidence_strength: str, allowed: Optional[list] = None) -> dict:
    """
    Heuristic extraction confidence is capped — keyword matches are weaker
    than an LLM read. evidence_strength: 'strong' (explicit phrase match),
    'medium' (sector-typical inference), 'weak' (default by sector).
    """
    if allowed is not None and value not in allowed:
        return _l1_unknown_field(name, "heuristic_invalid_enum")
    conf = {'strong': 0.70, 'medium': 0.55, 'weak': 0.40}.get(evidence_strength, 0.40)
    if conf < L1_MIN_FIELD_CONFIDENCE:
        return _l1_unknown_field(name, "heuristic_below_threshold")
    return _l1_field(value, conf, "heuristic")


def _heuristic_idea_features(idea_text: str, sector: str) -> dict:
    """Keyword-based fallback. Confidence is capped — heuristic ≠ LLM read."""
    t = idea_text.lower()
    seg_raw   = _infer_target_segment(t, sector)
    bm_raw    = _infer_business_model(t, sector, seg_raw)
    stage_raw = _infer_stage(t)
    diff_raw  = _infer_differentiation_score(t)

    # Evidence strength per field — strong only when explicit signal in text.
    # Weak when only sector-fallback (no in-text signal at all) — that's the
    # silent-default trap the audit flagged.
    explicit_seg   = any(tok in t for tok in ('b2b', 'b2c', 'b2g', 'enterprise', 'consumer'))
    explicit_bm    = any(tok in t for tok in ('subscription', 'marketplace', 'commission', 'saas', 'platform'))
    explicit_stage = any(tok in t for tok in ('mvp', 'beta', 'pilot', 'launched', 'live customers', 'mrr'))
    explicit_diff  = any(tok in t for tok in (
        'ai-powered', 'breakthrough', 'patent', 'proprietary',
        'similar to', 'clone', 'like uber', 'like amazon',
    ))

    # Implicit-signal channels — softer than explicit but still in-text, not sector-only.
    has_segment_signal = (
        _count_any(t, BUSINESS_CUSTOMER_HINTS) > 0
        or _count_any(t, CONSUMER_HINTS) > 0
    )
    has_bm_signal = (
        _has_any(t, SOFTWARE_HINTS)
        or _has_any(t, OPERATIONS_HINTS)
        or _has_any(t, RESTAURANT_HINTS)
        or _has_any(t, AGRITECH_STRONG_HINTS)
        or _has_any(t, SERVICE_HINTS)
        or _has_any(t, HARDWARE_HINTS)
        # Sector-anchoring operational tokens: when the text uses sector-typical
        # vocabulary (invoice, lending, payments, etc.), the business-model
        # inference is at least medium-evidence even without an explicit
        # "subscription"/"marketplace"-style word.
        or any(_phrase_in_text(t, kw) for kw in SECTOR_KEYWORDS.get(sector, []))
    )

    seg_strength = 'strong' if explicit_seg else ('medium' if has_segment_signal else 'weak')
    bm_strength  = 'strong' if explicit_bm  else ('medium' if has_bm_signal      else 'weak')
    stage_strength = 'strong' if explicit_stage else 'medium'  # 'idea' default is weak but acceptable

    seg   = _heuristic_field('target_segment',  seg_raw,   seg_strength,   TARGET_SEGMENTS)
    bm    = _heuristic_field('business_model',  bm_raw,    bm_strength,    BUSINESS_MODELS)
    stage = _heuristic_field('stage',           stage_raw, stage_strength, STAGES)
    diff  = _heuristic_field('differentiation_score', diff_raw, 'strong' if explicit_diff else 'weak')

    # Competitive intensity / regulatory risk — sector-typical inference (medium evidence)
    comp_val = {'fintech': 'high', 'ecommerce': 'high', 'healthtech': 'medium',
                'saas': 'medium', 'edtech': 'medium', 'logistics': 'medium',
                'agritech': 'low', 'other': 'medium'}.get(sector, 'medium')
    if _has_any(t, RESTAURANT_HINTS) and bm_raw == 'saas':
        comp_val = 'medium'
    reg_val = {'fintech': 'high', 'healthtech': 'high', 'edtech': 'low',
               'ecommerce': 'low', 'saas': 'low', 'logistics': 'medium',
               'agritech': 'low', 'other': 'medium'}.get(sector, 'medium')

    comp = _heuristic_field('competitive_intensity', comp_val, 'medium', INTENSITY_LEVELS)
    reg  = _heuristic_field('regulatory_risk',       reg_val,  'medium', INTENSITY_LEVELS)

    # Market readiness — only strong when text gives explicit pull/push signal
    ready_val = 3
    explicit_ready = False
    if any(w in t for w in ['proven market', 'large demand', 'everyone needs', 'mass market', 'high demand']):
        ready_val = 4; explicit_ready = True
    if any(w in t for w in ['pioneer', 'niche', 'emerging need', 'early adopters', 'creating the market']):
        ready_val = 2; explicit_ready = True
    if _has_any(t, ['cut waste', 'reduce waste', 'save money', 'reduce cost', 'working capital']):
        ready_val = max(ready_val, 4); explicit_ready = True
    ready = _heuristic_field('market_readiness', ready_val, 'medium' if explicit_ready else 'weak')

    # Heuristic monetization mirrors business_model (weaker signal)
    mon = _heuristic_field('monetization', bm_raw, 'medium' if explicit_bm else 'weak', MONETIZATION_TYPES)

    fields = {
        'business_model': bm,
        'target_segment': seg,
        'monetization':   mon,
        'stage':          stage,
        'differentiation_score': diff,
        'competitive_intensity': comp,
        'regulatory_risk':       reg,
        'market_readiness':      ready,
    }
    return _result_from_fields(fields, sector, idea_text)


def _backfill_with_heuristic(llm_result: dict, idea_text: str, sector: str) -> dict:
    """
    For required fields the LLM marked UNKNOWN, attempt a heuristic backfill
    ONLY if the heuristic has explicit-keyword evidence (confidence ≥ threshold).
    Never fills with low-evidence guesses — UNKNOWN must remain UNKNOWN otherwise.
    """
    h = _heuristic_idea_features(idea_text, sector)
    fields = {name: {"value": llm_result["values"][name],
                     "confidence": llm_result["confidence"][name],
                     "source": llm_result["source"][name]}
              for name in llm_result["values"]}

    for name in L1_REQUIRED_FIELDS:
        if name not in llm_result["unknown_required"]:
            continue
        h_conf = h["confidence"].get(name, 0.0)
        h_val  = h["values"].get(name)
        if h_conf >= L1_MIN_FIELD_CONFIDENCE and h_val not in (UNKNOWN_VALUE, None):
            fields[name] = _l1_field(h_val, h_conf, "heuristic")

    return _result_from_fields(fields, sector, idea_text)


def _validate_l1_consistency(values: dict, sector: str, idea_text: str) -> dict:
    """
    Cross-field schema integrity. Rejects logically incompatible combinations.
    UNKNOWN fields are skipped (already handled by the sufficiency gate).
    Returns {"ok": bool, "violations": [<reason codes>]}
    """
    violations: list = []
    bm  = values.get('business_model')
    seg = values.get('target_segment')
    mon = values.get('monetization')
    sec = (sector or '').lower()

    def _known(*xs):
        return all(x not in (UNKNOWN_VALUE, None) for x in xs)

    # Sector × business_model
    if _known(bm) and sec == 'agritech' and bm == 'saas' and 'farm' not in (idea_text or '').lower():
        violations.append("agritech_saas_without_farm_signal")
    if _known(bm) and sec == 'healthtech' and bm == 'marketplace' and seg == 'b2c':
        violations.append("healthtech_b2c_marketplace_regulatory_conflict")

    # Business_model × target_segment
    if _known(bm, seg) and bm == 'hardware' and seg == 'b2c' and mon == 'freemium':
        violations.append("hardware_b2c_freemium_unviable")
    if _known(bm, seg) and bm == 'saas' and seg == 'b2c' and mon == 'commission':
        violations.append("saas_b2c_commission_mismatch")

    # Monetization × segment
    if _known(seg, mon) and seg == 'b2g' and mon == 'ad-based':
        violations.append("b2g_ad_based_unviable")
    if _known(bm, mon) and bm == 'marketplace' and mon == 'subscription' and seg == 'b2c':
        violations.append("b2c_marketplace_subscription_unusual")

    return {"ok": not violations, "violations": violations}


def _l1_clarification_message(result: dict) -> dict:
    """Translate an insufficient L1 result into a user-facing controlled state."""
    asks = []
    if 'business_model' in result["unknown_required"]:
        asks.append("How does it make money — subscription, marketplace, commission, SaaS, or service?")
    if 'target_segment' in result["unknown_required"]:
        asks.append("Who pays — businesses (B2B), consumers (B2C), or government (B2G)?")
    if 'stage' in result["unknown_required"]:
        asks.append("Where are you — idea, validation, MVP shipped, or already growing?")
    if not asks and not result["consistency"]["ok"]:
        asks.append(
            "The combination of model, segment, and monetization you described is internally inconsistent — "
            "clarify which one is the anchor and the others should follow."
        )
    if not asks:
        asks.append("Restate the idea so the customer, the mechanism, and the revenue source are explicit.")
    return {
        "missing_fields":      result["unknown_required"],
        "consistency_issues":  result["consistency"]["violations"],
        "aggregate_confidence": result["aggregate_confidence"],
        "questions":           asks[:3],
    }




# Export everything defined in this module — including underscore-prefixed
# helpers — so other midan submodules can wildcard-import the full surface.
__all__ = [name for name in list(globals().keys()) if not name.startswith('__')]
