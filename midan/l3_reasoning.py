"""
midan.l3_reasoning — structured idea reasoning layer.

Five analyzers (differentiation, competition, business_model, unit_economics,
signal_interactions) each emit a structured envelope grounded in L1+L2
signals. No new scalar score introduced; legacy `idea_signal` preserved
for the L4 TAS contract but tagged legacy. Insufficient-information state
is explicit per analyzer.
"""
from midan.core import *  # noqa: F401,F403
from midan.l1_parser import UNKNOWN_VALUE  # foundational sentinel  # noqa: F401


# ── extracted from api.py ─────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════
# LAYER 3 — IDEA SIGNAL SCORING
# Encodes domain knowledge: which business models thrive in which regimes.
# This is the PRIMARY driver of output variability across different ideas.
# ═══════════════════════════════════════════════════════════════

# (regime, business_model, target_segment) → base fit score
# Different combinations produce meaningfully different base scores.
_FIT_TABLE: Dict[tuple, float] = {
    # GROWTH MARKET — aggressive expansion is rewarded
    ('GROWTH_MARKET', 'saas',         'b2b'):   0.91,
    ('GROWTH_MARKET', 'subscription', 'b2b'):   0.90,
    ('GROWTH_MARKET', 'marketplace',  'b2c'):   0.87,
    ('GROWTH_MARKET', 'marketplace',  'mixed'): 0.85,
    ('GROWTH_MARKET', 'commission',   'mixed'): 0.84,
    ('GROWTH_MARKET', 'commission',   'b2c'):   0.83,
    ('GROWTH_MARKET', 'subscription', 'b2c'):   0.80,
    ('GROWTH_MARKET', 'service',      'b2b'):   0.77,
    ('GROWTH_MARKET', 'hardware',     'mixed'): 0.72,
    ('GROWTH_MARKET', 'other',        'mixed'): 0.70,

    # EMERGING MARKET — subscription/SaaS most resilient; B2C needs patience
    ('EMERGING_MARKET', 'saas',         'b2b'):   0.84,
    ('EMERGING_MARKET', 'subscription', 'b2b'):   0.82,
    ('EMERGING_MARKET', 'marketplace',  'mixed'): 0.77,
    ('EMERGING_MARKET', 'commission',   'mixed'): 0.73,
    ('EMERGING_MARKET', 'commission',   'b2b'):   0.71,
    ('EMERGING_MARKET', 'subscription', 'b2c'):   0.67,
    ('EMERGING_MARKET', 'service',      'b2b'):   0.66,
    ('EMERGING_MARKET', 'marketplace',  'b2b'):   0.64,
    ('EMERGING_MARKET', 'service',      'b2c'):   0.55,
    ('EMERGING_MARKET', 'other',        'mixed'): 0.58,

    # HIGH FRICTION — only cost-saving B2B tools survive
    ('HIGH_FRICTION_MARKET', 'saas',         'b2b'):   0.72,
    ('HIGH_FRICTION_MARKET', 'subscription', 'b2b'):   0.68,
    ('HIGH_FRICTION_MARKET', 'commission',   'b2b'):   0.54,
    ('HIGH_FRICTION_MARKET', 'service',      'b2b'):   0.50,
    ('HIGH_FRICTION_MARKET', 'marketplace',  'mixed'): 0.43,
    ('HIGH_FRICTION_MARKET', 'commission',   'b2c'):   0.40,
    ('HIGH_FRICTION_MARKET', 'subscription', 'b2c'):   0.38,
    ('HIGH_FRICTION_MARKET', 'other',        'b2c'):   0.32,

    # CONTRACTING — survival mode; only essential B2B cost-cutters viable
    ('CONTRACTING_MARKET', 'saas',         'b2b'):   0.58,
    ('CONTRACTING_MARKET', 'service',      'b2b'):   0.52,
    ('CONTRACTING_MARKET', 'subscription', 'b2b'):   0.50,
    ('CONTRACTING_MARKET', 'commission',   'b2b'):   0.42,
    ('CONTRACTING_MARKET', 'marketplace',  'mixed'): 0.32,
    ('CONTRACTING_MARKET', 'marketplace',  'b2c'):   0.28,
    ('CONTRACTING_MARKET', 'subscription', 'b2c'):   0.25,
    ('CONTRACTING_MARKET', 'other',        'b2c'):   0.20,
}

_REGIME_DEFAULTS: Dict[str, float] = {
    'GROWTH_MARKET': 0.72, 'EMERGING_MARKET': 0.60,
    'HIGH_FRICTION_MARKET': 0.45, 'CONTRACTING_MARKET': 0.35,
}

# ── BM-CONDITIONAL WEIGHT PROFILES ────────────────────────────────────────────
# Different business model types have structurally different success drivers.
# A marketplace lives or dies on liquidity — competition is lethal (winner-takes-all).
# SaaS/Subscription succeeds through differentiation and switching cost creation.
# Commission/Fintech: regulatory risk is existential, not just a drag.
# Service: competition matters less; scalability is the binding constraint.
# Hardware: stage is the dominant penalty — capital requirements are brutal early.
#
# Applying uniform weights to all BM types produces meaningless signals.
# This table encodes domain knowledge about what actually drives each model.
_BM_PROFILE: Dict[str, dict] = {
    'marketplace': {
        'diff_scale':    0.07,   # differentiation matters LESS — supply/demand fit matters more
        'ready_scale':   0.08,   # market readiness matters MORE — need existing demand, not pioneers
        'stage_deltas':  {'idea': -0.14, 'validation': -0.06, 'mvp': +0.10, 'growth': +0.20},
        'comp_deltas':   {'low': +0.10, 'medium': 0.0, 'high': -0.16},  # winner-takes-all: competition lethal
        'dominant_risk': 'liquidity',
        'key_signal':    'market_readiness',
        'moat_source':   'network effects and supply lock-in',
    },
    'saas': {
        'diff_scale':    0.13,   # differentiation matters MORE — switching costs = moat
        'ready_scale':   0.04,
        'stage_deltas':  {'idea': -0.07, 'validation': -0.01, 'mvp': +0.08, 'growth': +0.15},
        'comp_deltas':   {'low': +0.07, 'medium': 0.0, 'high': -0.08},
        'dominant_risk': 'differentiation',
        'key_signal':    'differentiation_score',
        'moat_source':   'switching costs and workflow integration depth',
    },
    'subscription': {
        'diff_scale':    0.10,
        'ready_scale':   0.06,   # readiness drives recurring willingness-to-pay
        'stage_deltas':  {'idea': -0.08, 'validation': -0.02, 'mvp': +0.07, 'growth': +0.14},
        'comp_deltas':   {'low': +0.05, 'medium': 0.0, 'high': -0.07},
        'dominant_risk': 'churn',
        'key_signal':    'market_readiness',
        'moat_source':   'habit formation and cancellation friction',
    },
    'commission': {
        'diff_scale':    0.09,
        'ready_scale':   0.05,
        'stage_deltas':  {'idea': -0.10, 'validation': -0.03, 'mvp': +0.07, 'growth': +0.15},
        'comp_deltas':   {'low': +0.06, 'medium': 0.0, 'high': -0.11},
        'dominant_risk': 'regulatory',  # commission models attract regulatory scrutiny
        'key_signal':    'regulatory_risk',
        'moat_source':   'regulatory licensing once obtained',
    },
    'service': {
        'diff_scale':    0.09,
        'ready_scale':   0.04,
        'stage_deltas':  {'idea': -0.05, 'validation': +0.01, 'mvp': +0.06, 'growth': +0.12},  # services monetize earlier
        'comp_deltas':   {'low': +0.04, 'medium': 0.0, 'high': -0.05},   # competition less relevant for services
        'dominant_risk': 'scalability',
        'key_signal':    'differentiation_score',
        'moat_source':   'proprietary methodology and key person relationships',
    },
    'hardware': {
        'diff_scale':    0.12,
        'ready_scale':   0.05,
        'stage_deltas':  {'idea': -0.15, 'validation': -0.07, 'mvp': +0.06, 'growth': +0.17},  # capital-intensive
        'comp_deltas':   {'low': +0.07, 'medium': 0.0, 'high': -0.10},
        'dominant_risk': 'capital',
        'key_signal':    'differentiation_score',
        'moat_source':   'proprietary hardware design and manufacturing relationships',
    },
    'other': {
        'diff_scale':    0.11,
        'ready_scale':   0.04,
        'stage_deltas':  {'idea': -0.09, 'validation': -0.02, 'mvp': +0.06, 'growth': +0.14},
        'comp_deltas':   {'low': +0.06, 'medium': 0.0, 'high': -0.09},
        'dominant_risk': 'execution',
        'key_signal':    'differentiation_score',
        'moat_source':   'speed and execution quality',
    },
}

# Per-sector regulatory sensitivity — fintech and healthtech are in a different league
_SECTOR_REG_PROFILE: Dict[str, dict] = {
    'fintech': {
        'reg_deltas':  {'low': +0.03, 'medium': -0.05, 'high': -0.18},  # CBE licensing = existential
        'b2b_bonus':    0.05,   # B2B fintech less exposed to consumer protection
        'trust_factor': True,   # regulatory risk is also trust barrier — double exposure
    },
    'healthtech': {
        'reg_deltas':  {'low': +0.04, 'medium': -0.06, 'high': -0.20},  # medical approval = long, expensive
        'b2b_bonus':    0.07,   # hospital/clinic sales: clearer buyer, less consumer friction
        'trust_factor': True,
    },
    'ecommerce': {
        'reg_deltas':  {'low': +0.04, 'medium': 0.0, 'high': -0.08},
        'b2b_bonus':    0.0,
        'trust_factor': False,
    },
    'edtech': {
        'reg_deltas':  {'low': +0.04, 'medium': -0.01, 'high': -0.09},
        'b2b_bonus':    0.03,
        'trust_factor': False,
    },
    'saas': {
        'reg_deltas':  {'low': +0.05, 'medium': 0.0, 'high': -0.10},
        'b2b_bonus':    0.04,
        'trust_factor': False,
    },
    'logistics': {
        'reg_deltas':  {'low': +0.03, 'medium': -0.02, 'high': -0.12},
        'b2b_bonus':    0.03,
        'trust_factor': False,
    },
    '_default': {
        'reg_deltas':  {'low': +0.04, 'medium': 0.0, 'high': -0.11},
        'b2b_bonus':    0.0,
        'trust_factor': False,
    },
}


def compute_idea_signal(idea_features: dict, regime: str, sector: str = 'other', idea_text: str = '') -> dict:
    """
    Layer 3 — Context-aware idea signal scoring.

    v2 UPGRADE: Conditional logic by business model type.
    Each BM type has different dominant success factors:
    - Marketplace: liquidity/network effects; competition is lethal
    - SaaS: differentiation (switching costs = moat); highly weighted
    - Commission: regulatory exposure is the binding constraint
    - Subscription: market readiness drives recurring willingness-to-pay
    - Service: scalability is the constraint; competition matters less
    - Hardware: stage penalty is amplified (capital requirements)

    Additionally, sector-specific regulatory profiles:
    - Fintech: CBE/regulatory risk is existential, not just a drag
    - Healthtech: medical approval creates 2-3× longer timelines
    - Others: standard profile

    Returns: idea_signal (float 0–1), breakdown (dict), dominant_risk, key_signal, moat_source
    """
    bm    = idea_features.get('business_model',       'other')
    seg   = idea_features.get('target_segment',        'b2c')
    diff  = idea_features.get('differentiation_score', 3)
    stage = idea_features.get('stage',                 'idea')
    comp  = idea_features.get('competitive_intensity', 'medium')
    reg   = idea_features.get('regulatory_risk',       'medium')
    ready = idea_features.get('market_readiness',      3)

    # 1. Base fit from regime × model × segment table
    base = (
        _FIT_TABLE.get((regime, bm, seg))
        or _FIT_TABLE.get((regime, bm, 'mixed'))
        or _FIT_TABLE.get((regime, 'other', 'mixed'))
        or _REGIME_DEFAULTS.get(regime, 0.50)
    )

    # 2. Load BM-specific profile (conditional weights)
    bm_profile     = _BM_PROFILE.get(bm, _BM_PROFILE['other'])
    sector_profile = _SECTOR_REG_PROFILE.get(sector, _SECTOR_REG_PROFILE['_default'])

    # 3. Differentiation multiplier — scale varies by BM type
    diff_mult = 0.78 + (diff - 1) * bm_profile['diff_scale']

    # 4. Stage delta — varies by BM (hardware/marketplace more stage-sensitive)
    stage_delta = bm_profile['stage_deltas'].get(stage, 0.0)

    # 5. Competitive drag — varies by BM (winner-takes-all models more sensitive)
    comp_delta = bm_profile['comp_deltas'].get(comp, 0.0)

    # 6. Regulatory friction — varies by SECTOR (fintech/healthtech amplified)
    reg_delta = sector_profile['reg_deltas'].get(reg, 0.0)

    # B2B segment partial relief in high-regulatory sectors
    # (institutional buyers vs consumer protection exposure)
    if seg == 'b2b' and sector_profile['b2b_bonus'] > 0:
        b2b_relief = sector_profile['b2b_bonus'] * (0.4 if reg == 'high' else 0.6)
        reg_delta += b2b_relief

    # 7. Market readiness — scale varies by BM
    ready_delta = (ready - 3) * bm_profile['ready_scale']

    # 8. Compounded penalty: early-stage in hostile regime
    # In CONTRACTING/HIGH_FRICTION, early ideas face amplified survival risk
    if stage in ('idea', 'validation') and regime in ('CONTRACTING_MARKET', 'HIGH_FRICTION_MARKET'):
        stage_delta *= 1.35

    idea_signal = float(np.clip(
        base * diff_mult + stage_delta + comp_delta + reg_delta + ready_delta,
        0.12, 0.95
    ))

    breakdown = {
        'model_regime_fit':    round(base, 3),
        'differentiation':     round((diff_mult - 1.0) * base, 3),
        'stage_readiness':     round(stage_delta, 3),
        'competitive_density': round(comp_delta, 3),
        'regulatory_exposure': round(reg_delta, 3),
        'market_pull':         round(ready_delta, 3),
    }

    dominant_risk = bm_profile['dominant_risk']
    key_signal = bm_profile['key_signal']
    moat_source = bm_profile['moat_source']
    if bm == 'saas' and _is_workflow_software_idea((idea_text or '').lower(), seg):
        dominant_risk = 'workflow'
        key_signal = 'market_readiness'
        moat_source = 'deep workflow integration and provable ROI'

    return {
        'idea_signal':    idea_signal,
        'breakdown':      breakdown,
        'business_model': bm,
        'target_segment': seg,
        'diff_score':     diff,
        'stage':          stage,
        'dominant_risk':  dominant_risk,
        'key_signal':     key_signal,
        'moat_source':    moat_source,
    }


def _signal_tier(score: float) -> str:
    """Qualitative tier — replaces fake numeric precision."""
    if score >= 0.76: return "Strong"
    if score >= 0.60: return "Moderate"
    if score >= 0.44: return "Mixed"
    return "Weak"

# ═══════════════════════════════════════════════════════════════
# LAYER 3 — STRUCTURED REASONING (PRIMARY EXPLANATION LAYER)
#
# The scalar `idea_signal` produced above is preserved as a legacy contract
# for L4 TAS computation. The PRIMARY explanation is the structured reasoning
# envelope built here — every output is grounded in L1 fields and/or L2
# signals, every classification is paired with an explicit reasoning string,
# and insufficient inputs produce `available: false` with `missing` lists
# rather than guessed values.
#
# All static reference data below (sector baselines, competition map, BM
# templates, proxy tables, interaction rules) is heuristic and labeled as
# such in the response — it is NOT presented as observed market data.
# ═══════════════════════════════════════════════════════════════

# L3_REASONING_VERSION and L3_FIELD_CONFIDENCE_FLOOR are owned by midan.config
# and reach this module via `from midan.core import *`.

# ── Sector baseline mechanisms (HEURISTIC) ──────────────────────────────────
# Minimal list of "what a typical idea in this sector usually does." Used by
# the differentiation analyzer to compare against the incoming idea. These
# are intentionally short — not a knowledge ontology.
_SECTOR_BASELINE_MECHANISMS: Dict[str, list] = {
    'fintech':    ['KYC verification', 'transaction ledger', 'payment rails', 'basic risk scoring'],
    'healthtech': ['appointment booking', 'patient records', 'provider directory'],
    'edtech':     ['content delivery', 'progress tracking', 'assessments'],
    'ecommerce':  ['product catalog', 'checkout flow', 'logistics fulfillment'],
    'saas':       ['user dashboard', 'workflow automation', 'data export', 'auth and permissions'],
    'logistics':  ['route optimization', 'shipment tracking', 'driver dispatch'],
    'agritech':   ['crop monitoring', 'input procurement', 'price discovery'],
    'other':      ['core service delivery', 'customer onboarding'],
}

# Idea-text mechanism extraction. Each entry is a labeled mechanism with
# a small list of distinguishing keywords. A match means the idea is
# claiming this mechanism — useful for comparing against the baseline.
_MECHANISM_KEYWORDS: Dict[str, list] = {
    'AI/ML reasoning':        ['ai-powered', 'machine learning', 'ml model', 'predictive', 'deep learning', 'using ai'],
    'forecasting':            ['forecasting', 'demand forecasting', 'prediction', 'predict'],
    'embedded finance':       ['embedded finance', 'embedded payment', 'banking-as-a-service', 'baas'],
    'workflow integration':   ['erp integration', 'crm integration', 'workflow integration', 'integrate with'],
    'vertical specialization':['restaurant-specific', 'clinic-specific', 'industry-specific', 'vertical saas', 'niche to'],
    'alternative data':       ['alternative data', 'alt data', 'transaction data', 'behavioral data'],
    'automation':             ['automate', 'automation', 'auto-'],
    'real-time signal':       ['real-time', 'realtime', 'instant', 'on-demand'],
    'two-sided marketplace':  ['two-sided', 'multi-vendor', 'matching engine', 'buyers and sellers'],
    'invoice financing':      ['invoice financing', 'factoring', 'working capital'],
    'subscription delivery':  ['subscription', 'monthly plan', 'recurring revenue'],
}

# ── Sector competition map (HEURISTIC, sector-baseline only) ────────────────
# Direct = same mechanism players. Indirect = different mechanism, same job.
# Substitutes = non-product solutions users currently use today.
_SECTOR_COMPETITION_MAP: Dict[str, dict] = {
    'fintech': {
        'direct':      ['established fintech apps in this niche', 'neobank competitors', 'lending platforms'],
        'indirect':    ['traditional banks', 'microfinance institutions'],
        'substitutes': ['informal credit (family/friends)', 'savings circles (gameya/ROSCA)', 'employer payroll advance', 'cash transactions'],
    },
    'healthtech': {
        'direct':      ['existing health apps', 'telemedicine platforms'],
        'indirect':    ['hospital walk-ins', 'pharmacy consultations'],
        'substitutes': ['self-medication', 'family or community advice', 'searching online'],
    },
    'edtech': {
        'direct':      ['existing online learning platforms', 'tutoring marketplaces'],
        'indirect':    ['private tutors', 'in-person classes'],
        'substitutes': ['YouTube videos', 'free online courses', 'self-study with textbooks'],
    },
    'ecommerce': {
        'direct':      ['regional marketplaces', 'sector-specific online retailers'],
        'indirect':    ['social commerce on Instagram/WhatsApp', 'physical retail'],
        'substitutes': ['informal trade networks', 'cash-on-delivery direct sellers'],
    },
    'saas': {
        'direct':      ['established SaaS players in this category', 'open-source self-hosted alternatives'],
        'indirect':    ['horizontal tools used in adjacent categories'],
        'substitutes': ['Excel and spreadsheets', 'manual processes', 'in-house custom-built tools'],
    },
    'logistics': {
        'direct':      ['existing logistics platforms', 'fleet management software'],
        'indirect':    ['traditional 3PL providers', 'in-house logistics teams'],
        'substitutes': ['phone calls and WhatsApp coordination', 'paper-based dispatch'],
    },
    'agritech': {
        'direct':      ['existing agritech platforms'],
        'indirect':    ['agricultural co-operatives', 'extension services'],
        'substitutes': ['traditional farming knowledge', 'middleman networks', 'weather radio/news'],
    },
    'other': {
        'direct':      ['existing players in adjacent niches'],
        'indirect':    ['established service providers'],
        'substitutes': ['manual workarounds', 'doing nothing'],
    },
}

# ── BM money-flow + cost-structure templates (HEURISTIC) ────────────────────
_BM_MONEY_FLOW_TEMPLATES: Dict[str, dict] = {
    'saas': {
        'who_pays': 'the team or organization adopting the tool',
        'what_for': 'ongoing access to the platform and continued feature delivery',
        'when':     'recurring (monthly or annual subscription)',
    },
    'marketplace': {
        'who_pays': 'the side capturing the most value per transaction (often the demand side)',
        'what_for': 'matching of supply with demand and transaction enablement',
        'when':     'per-transaction (commission take rate)',
    },
    'subscription': {
        'who_pays': 'end users or organizations',
        'what_for': 'recurring delivery of a service or content',
        'when':     'monthly or annual cycles',
    },
    'commission': {
        'who_pays': 'the side gaining liquidity or completing the transaction',
        'what_for': 'each completed transaction or referral',
        'when':     'at point of transaction',
    },
    'service': {
        'who_pays': 'clients on engagement basis',
        'what_for': 'specific deliverables or hours billed',
        'when':     'on engagement, per project, or retained',
    },
    'hardware': {
        'who_pays': 'buyers acquiring the device',
        'what_for': 'one-time device purchase, optionally with service contract',
        'when':     'upfront with optional recurring service revenue',
    },
    'other': {
        'who_pays': 'unspecified — the idea description does not state the payer',
        'what_for': 'unspecified',
        'when':     'unspecified',
    },
}

_BM_COST_STRUCTURE_TEMPLATES: Dict[str, dict] = {
    'saas': {
        'fixed_cost_drivers':    ['engineering team', 'cloud infrastructure', 'product development'],
        'variable_cost_drivers': ['customer acquisition', 'support per user'],
        'operational_complexity': 'medium',
    },
    'marketplace': {
        'fixed_cost_drivers':    ['platform engineering', 'trust and safety operations'],
        'variable_cost_drivers': ['supply-side acquisition', 'demand-side acquisition', 'transaction support'],
        'operational_complexity': 'high',
    },
    'subscription': {
        'fixed_cost_drivers':    ['content or delivery infrastructure'],
        'variable_cost_drivers': ['acquisition spend', 'churn management and retention'],
        'operational_complexity': 'medium',
    },
    'commission': {
        'fixed_cost_drivers':    ['platform infrastructure', 'compliance and licensing'],
        'variable_cost_drivers': ['transaction processing', 'fraud handling'],
        'operational_complexity': 'medium',
    },
    'service': {
        'fixed_cost_drivers':    ['core team retention'],
        'variable_cost_drivers': ['delivery hours', 'project-specific resources'],
        'operational_complexity': 'high',
    },
    'hardware': {
        'fixed_cost_drivers':    ['R&D and tooling'],
        'variable_cost_drivers': ['component costs', 'manufacturing', 'logistics', 'returns'],
        'operational_complexity': 'high',
    },
    'other': {
        'fixed_cost_drivers':    [],
        'variable_cost_drivers': [],
        'operational_complexity': 'unknown',
    },
}

# ── Unit economics qualitative proxies (assumption-tier with reasoning) ─────
_CAC_PROXY_RULES: Dict[tuple, tuple] = {
    ('saas', 'b2b'):         ('medium', 'B2B SaaS sales cycles are structured but require multi-stakeholder approval.'),
    ('saas', 'b2c'):         ('high',   'B2C SaaS competes against free alternatives; CAC absorbs that pressure.'),
    ('saas', 'b2g'):         ('high',   'B2G procurement cycles are long and relationship-heavy.'),
    ('marketplace', 'b2c'):  ('high',   'Two-sided cold-start makes both supply and demand acquisition expensive.'),
    ('marketplace', 'b2b'):  ('medium', 'B2B marketplaces have fewer but higher-LTV participants.'),
    ('subscription', 'b2c'): ('high',   'Consumer subscription churn forces continuous re-acquisition.'),
    ('subscription', 'b2b'): ('medium', 'B2B subscription retention is structurally higher when value is clear.'),
    ('commission', 'b2c'):   ('medium', 'Commission models capture per-transaction value, supporting performance marketing.'),
    ('commission', 'b2b'):   ('medium', 'B2B commission needs trust upfront but compounds with each repeat transaction.'),
    ('service', 'b2b'):      ('low',    'B2B services often grow via referral and inbound from delivery quality.'),
    ('service', 'b2c'):      ('medium', 'Consumer services compete with substitutes; acquisition needs trust.'),
    ('hardware', 'b2c'):     ('high',   'Physical product CAC includes channel costs, returns, and trial friction.'),
    ('hardware', 'b2b'):     ('medium', 'B2B hardware sells through fewer, longer-cycle deals.'),
}

_RPU_PROXY_RULES: Dict[tuple, tuple] = {
    ('subscription', 'b2b'): ('high',   'Recurring B2B contracts compound annual contract value over time.'),
    ('subscription', 'b2c'): ('low',    'Consumer subscriptions are pressure-tested by price-sensitive churn.'),
    ('commission', 'b2b'):   ('medium', 'Commission scales with transaction value, which is higher in B2B.'),
    ('commission', 'b2c'):   ('low',    'Consumer transactions are smaller; many are needed for meaningful ARPU.'),
    ('one-time', 'b2c'):     ('low',    'Single transaction captures one moment of willingness-to-pay.'),
    ('one-time', 'b2b'):     ('medium', 'B2B one-time deals can be large but are non-recurring.'),
    ('freemium', 'b2c'):     ('low',    'Conversion rates from free tier are typically 1-5%; ARPU heavily diluted.'),
    ('freemium', 'b2b'):     ('medium', 'B2B freemium converts at higher rates when wedged into team workflows.'),
    ('ad-based', 'b2c'):     ('low',    'Advertising RPU is among the lowest monetization paths.'),
    ('ad-based', 'b2b'):     ('low',    'B2B advertising has narrow audiences and limited inventory.'),
}

_SCALABILITY_PROXY_RULES: Dict[str, tuple] = {
    'saas':         ('improves_with_scale',         'Software marginal cost is near-zero; gross margin expands as user count grows.'),
    'subscription': ('improves_with_scale',         'Recurring revenue model scales naturally if churn is bounded.'),
    'marketplace':  ('improves_after_liquidity',    'Operational cost is high until a liquidity threshold; after that, network effects amortize.'),
    'commission':   ('improves_with_scale',         'Take-rate revenue scales with transaction volume at limited incremental cost.'),
    'service':      ('worsens_with_scale',          'Service delivery is human-bound; growth requires linear hiring.'),
    'hardware':     ('worsens_with_scale',          'Physical supply chain adds working capital pressure with each unit.'),
    'other':        ('unknown',                     'Business model not specific enough to assess scalability pressure.'),
}


def _safe_int(v, default=3):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _l3_field_known(values: dict, confidence: dict, field: str) -> bool:
    """A field is usable for L3 reasoning iff it is non-UNKNOWN AND its confidence ≥ floor."""
    v = values.get(field)
    c = confidence.get(field, 0.0)
    if v in (UNKNOWN_VALUE, None):
        return False
    return c >= L3_FIELD_CONFIDENCE_FLOOR


# ── ANALYZER 1: Differentiation ─────────────────────────────────────────────
def _extract_idea_mechanisms(idea_text: str) -> list:
    """
    Returns the list of mechanism labels keyword-matched in idea_text.
    This is a deliberately small, transparent extraction — not NLU.
    """
    t = (idea_text or "").lower()
    matches = []
    for label, kws in _MECHANISM_KEYWORDS.items():
        if any(kw in t for kw in kws):
            matches.append(label)
    return matches


def _analyze_differentiation(idea_text: str, values: dict, confidence: dict, sector: str) -> dict:
    """
    Compare the idea's claimed mechanism (from text) to the sector baseline.
    Returns structured reasoning, not a score. If there is no usable sector
    signal, return available=false with missing fields rather than guess.
    """
    grounded_in = []
    if sector and sector != 'other':
        grounded_in.append('sector')

    sector_key = sector if sector in _SECTOR_BASELINE_MECHANISMS else 'other'
    baseline   = _SECTOR_BASELINE_MECHANISMS[sector_key]

    idea_mechanisms = _extract_idea_mechanisms(idea_text)
    diff_score      = _safe_int(values.get('differentiation_score'), 3) if _l3_field_known(values, confidence, 'differentiation_score') else None
    if diff_score is not None:
        grounded_in.append('differentiation_score')

    # what_is_new = mechanisms claimed in text that are NOT in the baseline
    what_is_new = [m for m in idea_mechanisms if m not in baseline]
    # what_is_standard = baseline items that align with claimed mechanisms (semantic check is heuristic — keyword overlap)
    what_is_standard = [b for b in baseline if any(b.split()[0] in idea_text.lower() for _ in [0])]
    # If text mentions baseline items by keyword overlap, surface them; else assume baseline is implicit
    text_lower = (idea_text or "").lower()
    what_is_standard = [b for b in baseline if any(tok in text_lower for tok in b.lower().split() if len(tok) > 4)]
    if not what_is_standard:
        # Treat baseline as implicit when nothing matches — surface that explicitly.
        what_is_standard_note = "baseline mechanisms assumed implicitly (not stated in idea text)"
    else:
        what_is_standard_note = "baseline mechanisms aligned with idea text"

    # what_is_missing = baseline items neither claimed nor mentioned
    what_is_missing = [b for b in baseline if b not in what_is_standard]

    # Verdict tier — derived strictly from list sizes + L1 diff score, not a new scalar
    if not idea_mechanisms and (diff_score is None or diff_score <= 2):
        verdict = "thin"
    elif len(what_is_new) >= 2 or (diff_score is not None and diff_score >= 4):
        verdict = "structural"
    else:
        verdict = "moderate"

    insufficient = (sector == 'other' and not idea_mechanisms)
    if insufficient:
        return {
            "available": False,
            "missing":   ["sector signal too weak for baseline comparison",
                          "no recognized mechanism keywords in idea text"],
            "reason":    "Differentiation analysis requires either a known sector or a mechanism keyword match in the idea text.",
            "evidence_grounded_in": {"l1_fields_used": grounded_in, "l2_fields_used": []},
        }

    return {
        "available":            True,
        "mechanism_extracted":  idea_mechanisms,
        "mechanism_extraction_method": "keyword_match",
        "sector_baseline":      {"sector": sector_key, "items": baseline, "source": "sector_baseline_heuristic"},
        "what_is_new":          [{"item": m, "source": "idea_inferred"} for m in what_is_new],
        "what_is_standard":     [{"item": s, "source": "sector_baseline"} for s in what_is_standard],
        "what_is_standard_note": what_is_standard_note,
        "what_is_missing":      [{"item": m, "source": "sector_baseline"} for m in what_is_missing],
        "verdict":              verdict,
        "verdict_reasoning":    (
            f"Idea claims {len(idea_mechanisms)} mechanism(s) above the sector baseline; "
            f"L1 differentiation_score={diff_score if diff_score is not None else 'unknown'}. "
            f"Verdict tier '{verdict}' is derived from these inputs, not from a new scalar."
        ),
        "evidence_grounded_in": {"l1_fields_used": grounded_in, "l2_fields_used": []},
    }


# ── ANALYZER 2: Competition ─────────────────────────────────────────────────
def _analyze_competition(idea_text: str, values: dict, confidence: dict, sector: str) -> dict:
    """
    Three-class competitive view: direct, indirect, substitutes. Sector
    items tagged source="sector_baseline"; idea-text named competitors
    tagged source="idea_inferred". Pressure tier uses L1 competitive_intensity.
    """
    grounded_in = []
    if sector in _SECTOR_COMPETITION_MAP:
        grounded_in.append('sector')

    if sector not in _SECTOR_COMPETITION_MAP and not _l3_field_known(values, confidence, 'business_model'):
        return {
            "available": False,
            "missing":   ["sector", "business_model"],
            "reason":    "Competition analysis requires either a known sector or a high-confidence business_model.",
            "evidence_grounded_in": {"l1_fields_used": grounded_in, "l2_fields_used": []},
        }

    sector_key = sector if sector in _SECTOR_COMPETITION_MAP else 'other'
    s_map      = _SECTOR_COMPETITION_MAP[sector_key]

    direct = [{"description": d, "source": "sector_baseline"} for d in s_map['direct']]
    indirect = [{"description": d, "source": "sector_baseline"} for d in s_map['indirect']]
    substitutes = [{"description": d, "source": "sector_baseline"} for d in s_map['substitutes']]

    # Idea-inferred named competitors: scan idea_text for "like X", "similar to X",
    # "compete with X", "alternative to X". Take the first 1–3 tokens after the
    # marker — that's the named entity. Stop at any connector word.
    text = (idea_text or "")
    inferred = []
    _CONNECTORS = {'for', 'to', 'in', 'with', 'and', 'or', 'but', 'on', 'at', 'by'}
    for marker in ['like ', 'similar to ', 'compete with ', 'alternative to ']:
        idx = text.lower().find(marker)
        if idx < 0:
            continue
        tail = text[idx + len(marker):].split('.')[0].split(',')[0].strip()
        if not tail:
            continue
        # Take up to the first connector word, max 4 tokens.
        name_tokens = []
        for tok in tail.split():
            stripped = tok.lower().strip('.,;:!?')
            if stripped in _CONNECTORS or len(name_tokens) >= 4:
                break
            name_tokens.append(tok)
        name = " ".join(name_tokens).strip()
        if name and 1 <= len(name) <= 60:
            inferred.append({
                "description": name,
                "source":      "idea_inferred",
                "reasoning":   f"Mentioned in idea text via '{marker.strip()}' phrase.",
            })
    direct = direct + inferred  # named competitors are direct by default

    # Pressure tier — strictly from L1 competitive_intensity if known
    comp = values.get('competitive_intensity')
    comp_known = _l3_field_known(values, confidence, 'competitive_intensity')
    if comp_known:
        grounded_in.append('competitive_intensity')
        pressure = comp
        pressure_reasoning = (
            f"L1 competitive_intensity={comp} (confidence "
            f"{confidence.get('competitive_intensity', 0.0):.2f}). "
            "Pressure tier uses this directly; pool sizes inform pool composition only."
        )
    else:
        pressure = "unknown"
        pressure_reasoning = "L1 competitive_intensity is UNKNOWN or below confidence floor."

    return {
        "available":           True,
        "direct_competitors":  direct,
        "indirect_competitors": indirect,
        "substitutes":         substitutes,
        "competitive_pressure": pressure,
        "competitive_pressure_reasoning": pressure_reasoning,
        "source_distribution": {
            "sector_baseline_count": sum(1 for d in direct + indirect + substitutes if d['source'] == 'sector_baseline'),
            "idea_inferred_count":   sum(1 for d in direct + indirect + substitutes if d['source'] == 'idea_inferred'),
        },
        "data_basis":          "heuristic_sector_map_plus_text_extraction",
        "evidence_grounded_in": {"l1_fields_used": grounded_in, "l2_fields_used": []},
    }


# ── ANALYZER 3: Business model ──────────────────────────────────────────────
def _analyze_business_model(values: dict, confidence: dict, sector: str) -> dict:
    """
    Money flow (who_pays, what_for, when) + cost structure (fixed, variable,
    operational_complexity) + viability reasoning. Templates are heuristic
    per-BM defaults, specialized with the L1 segment when available.
    """
    grounded_in = []
    if not _l3_field_known(values, confidence, 'business_model'):
        return {
            "available": False,
            "missing":   ["business_model"],
            "reason":    "Business model analysis requires a high-confidence business_model field.",
            "evidence_grounded_in": {"l1_fields_used": grounded_in, "l2_fields_used": []},
        }
    bm = values['business_model']
    grounded_in.append('business_model')

    money_flow = dict(_BM_MONEY_FLOW_TEMPLATES.get(bm, _BM_MONEY_FLOW_TEMPLATES['other']))
    seg = values.get('target_segment')
    if _l3_field_known(values, confidence, 'target_segment'):
        grounded_in.append('target_segment')
        money_flow['who_pays'] = f"{seg.upper()} {money_flow['who_pays']}"
    cost = dict(_BM_COST_STRUCTURE_TEMPLATES.get(bm, _BM_COST_STRUCTURE_TEMPLATES['other']))

    # Viability reasoning — references only the fields we have
    viability_parts = [
        f"Money flow: {money_flow['who_pays']} pay for {money_flow['what_for']} on a "
        f"{money_flow['when']} basis."
    ]
    if cost['fixed_cost_drivers'] or cost['variable_cost_drivers']:
        viability_parts.append(
            f"Cost shape: fixed costs come from {', '.join(cost['fixed_cost_drivers']) or 'unspecified'}; "
            f"variable costs come from {', '.join(cost['variable_cost_drivers']) or 'unspecified'}. "
            f"Operational complexity is {cost['operational_complexity']}."
        )
    if sector in ('fintech', 'healthtech') and _l3_field_known(values, confidence, 'regulatory_risk'):
        grounded_in.append('regulatory_risk')
        viability_parts.append(
            f"Sector ({sector}) imposes additional compliance overhead on top of the BM cost structure."
        )

    return {
        "available":            True,
        "label":                bm,
        "money_flow":           money_flow,
        "cost_structure":       cost,
        "viability_reasoning":  " ".join(viability_parts),
        "data_basis":           "heuristic_per_bm_template",
        "evidence_grounded_in": {"l1_fields_used": grounded_in, "l2_fields_used": []},
    }


# ── ANALYZER 4: Unit economics (qualitative proxies) ────────────────────────
def _analyze_unit_economics(values: dict, confidence: dict, sector: str) -> dict:
    """
    Qualitative CAC, RPU, scalability_pressure tiers — explicitly framed as
    assumption-tier proxies, not numerical estimates.
    """
    grounded_in = []
    if not (_l3_field_known(values, confidence, 'business_model')
            and _l3_field_known(values, confidence, 'target_segment')):
        return {
            "available": False,
            "missing":   [f for f in ('business_model', 'target_segment')
                          if not _l3_field_known(values, confidence, f)],
            "reason":    "Unit economics proxies require both business_model and target_segment at confidence ≥ 0.55.",
            "evidence_grounded_in": {"l1_fields_used": grounded_in, "l2_fields_used": []},
        }
    bm  = values['business_model']
    seg = values['target_segment']
    grounded_in += ['business_model', 'target_segment']

    cac_tier, cac_reasoning = _CAC_PROXY_RULES.get((bm, seg), ('unknown', 'No CAC proxy mapped for this BM × segment combination.'))

    mon = values.get('monetization')
    if _l3_field_known(values, confidence, 'monetization'):
        grounded_in.append('monetization')
        rpu_tier, rpu_reasoning = _RPU_PROXY_RULES.get((mon, seg), ('unknown', f'No RPU proxy mapped for monetization={mon} × {seg}.'))
    else:
        rpu_tier, rpu_reasoning = 'unknown', 'L1 monetization is UNKNOWN — RPU proxy cannot be derived.'

    scal_tier, scal_reasoning = _SCALABILITY_PROXY_RULES.get(bm, _SCALABILITY_PROXY_RULES['other'])

    return {
        "available":             True,
        "assumption_basis":      "qualitative_proxy",
        "framing_caveat":        "These are qualitative tier assumptions, not factual estimates. Used for reasoning, not for financial planning.",
        "cac_proxy":             {"tier": cac_tier, "reasoning": cac_reasoning},
        "revenue_per_user_proxy":{"tier": rpu_tier, "reasoning": rpu_reasoning},
        "scalability_pressure":  {"tier": scal_tier, "reasoning": scal_reasoning},
        "evidence_grounded_in":  {"l1_fields_used": grounded_in, "l2_fields_used": []},
    }


# ── ANALYZER 5: Signal interactions (explicit rule set) ─────────────────────
# Each rule is gated on its involved L1 fields being non-UNKNOWN at
# confidence ≥ L3_FIELD_CONFIDENCE_FLOOR. Predicates take (values, regime).
_SIGNAL_INTERACTION_RULES: list = [
    {
        'id':                'low_diff_high_competition',
        'involved_signals':  ['differentiation_score', 'competitive_intensity'],
        'predicate':         lambda v, regime: _safe_int(v.get('differentiation_score'), 3) <= 2 and v.get('competitive_intensity') == 'high',
        'consequence':       'amplified_competitive_risk',
        'severity':          'high',
        'explanation':       'Low differentiation in a high-competition space is structurally fragile — incumbents can replicate the offering and out-distribute it before traction compounds.',
    },
    {
        'id':                'strong_fit_weak_market',
        'involved_signals':  ['market_readiness'],
        'predicate':         lambda v, regime: regime in ('GROWTH_MARKET', 'EMERGING_MARKET') and _safe_int(v.get('market_readiness'), 3) <= 2,
        'consequence':       'limited_upside',
        'severity':          'medium',
        'explanation':       'Favorable sector regime with low market readiness means the macro is supportive but customer demand has not yet formed — the upside is theoretical until pull signals emerge.',
    },
    {
        'id':                'early_stage_high_regulatory',
        'involved_signals':  ['stage', 'regulatory_risk'],
        'predicate':         lambda v, regime: v.get('stage') in ('idea', 'validation') and v.get('regulatory_risk') == 'high',
        'consequence':       'regulatory_validation_burden',
        'severity':          'high',
        'explanation':       'Pre-MVP idea facing high regulatory risk: validation is gated by licensing or compliance lead time, not just product-market fit.',
    },
    {
        'id':                'b2c_high_regulatory',
        'involved_signals':  ['target_segment', 'regulatory_risk'],
        'predicate':         lambda v, regime: v.get('target_segment') == 'b2c' and v.get('regulatory_risk') == 'high',
        'consequence':       'consumer_protection_double_exposure',
        'severity':          'medium',
        'explanation':       'B2C with high regulatory risk faces dual exposure: regulator scrutiny PLUS consumer-protection-driven churn when issues surface.',
    },
    {
        'id':                'growth_stage_low_diff',
        'involved_signals':  ['stage', 'differentiation_score'],
        'predicate':         lambda v, regime: v.get('stage') == 'growth' and _safe_int(v.get('differentiation_score'), 3) <= 2,
        'consequence':       'valuation_ceiling',
        'severity':          'medium',
        'explanation':       'Growth-stage with thin differentiation: scaling exposes the lack of moat, capping valuation multiples and inviting margin compression.',
    },
    {
        'id':                'hardware_early_stage',
        'involved_signals':  ['business_model', 'stage'],
        'predicate':         lambda v, regime: v.get('business_model') == 'hardware' and v.get('stage') in ('idea', 'validation'),
        'consequence':       'capital_burn_risk',
        'severity':          'high',
        'explanation':       'Hardware models pre-MVP carry capital burn risk that software pivots cannot escape — every iteration costs unit economics, not just engineering time.',
    },
]


def _analyze_signal_interactions(values: dict, confidence: dict, regime: str) -> list:
    """
    Run each interaction rule. A rule fires only if ALL its involved L1
    signals are usable (non-UNKNOWN, confidence ≥ floor). Each fired
    interaction is returned with its full trace.
    """
    fired: list = []
    for rule in _SIGNAL_INTERACTION_RULES:
        # Confirm every involved L1 signal is usable. (regime is L2 — always present.)
        l1_signals = [s for s in rule['involved_signals'] if s != 'regime']
        if not all(_l3_field_known(values, confidence, f) for f in l1_signals):
            continue
        try:
            if not rule['predicate'](values, regime):
                continue
        except Exception as _pred_err:
            __import__('logging').getLogger("midan.l3").warning(
                "[L3] interaction-rule predicate '%s' raised (%s: %r) — rule skipped",
                rule.get('id', '?'), type(_pred_err).__name__, _pred_err,
            )
            continue
        fired.append({
            'interaction_id':    rule['id'],
            'involved_signals':  rule['involved_signals'],
            'consequence':       rule['consequence'],
            'severity':          rule['severity'],
            'explanation':       rule['explanation'],
            'evidence_grounded_in': {
                'l1_fields_used': l1_signals,
                'l2_fields_used': ['regime'] if 'regime' in rule['involved_signals'] else [],
            },
        })
    return fired


# ── ORCHESTRATOR ────────────────────────────────────────────────────────────
def compute_l3_reasoning(
    idea_text: str,
    l1_result: dict,
    regime: str,
    sector: str,
    idea_signal_data: dict,
) -> dict:
    """
    Build the structured L3 reasoning envelope. This is the PRIMARY
    explanation surface for L3; the legacy scalar `idea_signal` from
    `compute_idea_signal` is preserved separately for L4 TAS.
    """
    values     = (l1_result or {}).get("values", {})
    confidence = (l1_result or {}).get("confidence", {})

    diff_analysis  = _analyze_differentiation(idea_text, values, confidence, sector)
    comp_analysis  = _analyze_competition(idea_text, values, confidence, sector)
    bm_analysis    = _analyze_business_model(values, confidence, sector)
    ue_analysis    = _analyze_unit_economics(values, confidence, sector)
    interactions   = _analyze_signal_interactions(values, confidence, regime)

    insufficient = []
    for module_name, module in [
        ('differentiation', diff_analysis),
        ('competition',     comp_analysis),
        ('business_model',  bm_analysis),
        ('unit_economics',  ue_analysis),
    ]:
        if not module.get('available', True):
            insufficient.append({
                'module':         module_name,
                'missing_fields': module.get('missing', []),
                'reason':         module.get('reason', 'required L1 fields are UNKNOWN or below confidence threshold'),
            })

    return {
        'reasoning_layer_version':  L3_REASONING_VERSION,
        'is_primary_explanation':   True,
        'differentiation':          diff_analysis,
        'competition':              comp_analysis,
        'business_model':           bm_analysis,
        'unit_economics':           ue_analysis,
        'signal_interactions':      interactions,
        'insufficient_information': insufficient,
        'legacy_scalar_signal':     {
            'value':   round(float(idea_signal_data.get('idea_signal', 0.0)), 3),
            'note':    'Preserved for L4 TAS computation. Not the primary explanation — use the structured fields above.',
        },
        'data_provenance': {
            'sector_baselines_source':   'heuristic_static_table',
            'competition_map_source':    'heuristic_static_table',
            'bm_templates_source':       'heuristic_static_table',
            'unit_economics_source':     'qualitative_proxy_rules',
            'interaction_rules_source':  'explicit_rule_set',
            'live_data_integration':     False,
        },
    }




# Export everything defined in this module — including underscore-prefixed
# helpers — so other midan submodules can wildcard-import the full surface.
__all__ = [name for name in list(globals().keys()) if not name.startswith('__')]
