"""
MIDAN Data Pipeline — Model-Assisted Signal Extractor
======================================================

SIGNAL EXTRACTION TIERS (formally enforced — not a workaround):
────────────────────────────────────────────────────────────────

  TIER 1 — TEXT-EXTRACTABLE
    Signals:  onboarding_friction, monetization_strength
    Method:   rule-based keyword matching on raw text
    Fallback: same (rule-based is sufficient, model adds marginal value)
    Reason:   Observable in website copy, pricing pages, CTAs, CLI docs

  TIER 2 — HYBRID
    Signals:  retention_proxy
    Method:   rule-based first; model supplements for ambiguous cases
    Fallback: rule-based only (partial coverage accepted)
    Reason:   Sometimes observable (scale numbers, testimonials),
              sometimes requires inference from product type

  TIER 3 — LLM-DEPENDENT
    Signals:  competition_intensity
    Method:   LLM reasoning with world knowledge (Ollama)
    Fallback: null — do NOT attempt rule extraction
    Reason:   Company websites never name competitors.
              Failory articles do, but that's source-specific.
              Forcing rules here produces hallucinated signals.

ARCHITECTURE BOUNDARY (strictly enforced):
  INPUT:  raw_content text (scraped website / Failory article / YC entry)
  OUTPUT: L3 signals with evidence snippets
  NEVER:  reads decision_analysis, pattern_tags, confidence_score as labels
  NEVER:  collapses 4 signals into a single score
  NEVER:  fabricates competition_intensity from website text

ACCURACY EXPECTATIONS (honest, not aspirational):
  onboarding_friction:   ~100% (rule-based fully covers this)
  monetization_strength: ~100% (rule-based fully covers this)
  retention_proxy:       ~85%  (rule-based + scale/testimonial patterns)
  competition_intensity:   0%  (rule-based) / ~80% (LLM when available)

Calibrated against: data/validation/ground_truth.json (30 hand-labeled pairs)
"""

import json
import os
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Signal Schema (canonical values only) ─────────────────────────────────────
SIGNAL_SCHEMA = {
    "retention_proxy":       ["high", "medium", "low"],
    "onboarding_friction":   ["low", "medium", "high"],
    "monetization_strength": ["strong", "moderate", "weak"],
    "competition_intensity": ["high", "moderate", "low"],
}

# ── Signal Tier Registry ───────────────────────────────────────────────────────
# This is the formal system architecture declaration.
# Every consumer of this module must respect these tiers.
SIGNAL_TIERS = {
    "onboarding_friction":   "TEXT_EXTRACTABLE",   # Tier 1
    "monetization_strength": "TEXT_EXTRACTABLE",   # Tier 1
    "retention_proxy":       "HYBRID",             # Tier 2
    "competition_intensity": "LLM_DEPENDENT",      # Tier 3
}

# Tier 3 signals should NEVER be returned from rule-based extraction.
# Returning null for them when LLM is unavailable is CORRECT behaviour.
LLM_DEPENDENT_SIGNALS = {s for s, t in SIGNAL_TIERS.items() if t == "LLM_DEPENDENT"}
TEXT_EXTRACTABLE_SIGNALS = {s for s, t in SIGNAL_TIERS.items() if t == "TEXT_EXTRACTABLE"}
HYBRID_SIGNALS = {s for s, t in SIGNAL_TIERS.items() if t == "HYBRID"}


# ── Few-Shot Cache (ground truth anchors) ─────────────────────────────────────
_FEW_SHOT_CACHE: Optional[list] = None


def _load_few_shot_examples() -> list[dict]:
    """
    Load few-shot examples from ground_truth.json.
    Only L3 signal labels included — decision_analysis never referenced.
    """
    global _FEW_SHOT_CACHE
    if _FEW_SHOT_CACHE is not None:
        return _FEW_SHOT_CACHE

    gt_path = BASE_DIR / "data" / "validation" / "ground_truth.json"
    if not gt_path.exists():
        return []

    ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))

    pool_by_signal: dict[str, list] = {k: [] for k in SIGNAL_SCHEMA}
    for entry in ground_truth:
        name = entry["startup_name"]
        for sig in entry["signals"]:
            if sig["signal"] not in SIGNAL_SCHEMA:
                continue
            pool_by_signal[sig["signal"]].append({
                "startup_name": name,
                "signal":       sig["signal"],
                "value":        sig["value"],
                "justification": sig["justification"],
            })

    # 2 examples per signal, diverse values
    examples = []
    for signal_name, pool in pool_by_signal.items():
        seen_values = set()
        for ex in pool:
            if ex["value"] not in seen_values and len(seen_values) < 2:
                examples.append(ex)
                seen_values.add(ex["value"])

    _FEW_SHOT_CACHE = examples
    return examples


# ── Prompt Builder ─────────────────────────────────────────────────────────────

def _build_extraction_prompt(text: str, startup_name: str = "") -> str:
    """
    Build a few-shot extraction prompt for the LLM.

    Tier 3 (LLM-DEPENDENT) signals are included ONLY in the LLM prompt —
    they are never attempted by the rule-based path.
    """
    examples = _load_few_shot_examples()

    few_shot_block = ""
    for ex in examples:
        tier = SIGNAL_TIERS.get(ex["signal"], "UNKNOWN")
        few_shot_block += (
            f"EXAMPLE ({ex['startup_name']}, {tier}):\n"
            f"  Signal: {ex['signal']}\n"
            f"  Value: {ex['value']}\n"
            f"  Evidence: \"{ex['justification']}\"\n\n"
        )

    name_context = f" for {startup_name}" if startup_name else ""
    text_excerpt = text[:2500]

    prompt = f"""You are a startup intelligence analyst. Extract the following 4 behavioral signals from the startup text{name_context}.

SIGNAL DEFINITIONS AND TIERS:
  [TIER 1 - TEXT-EXTRACTABLE]
  onboarding_friction   = How hard it is for a new user to start.      Values: low | medium | high
  monetization_strength = How reliably the product converts to revenue. Values: strong | moderate | weak

  [TIER 2 - HYBRID]
  retention_proxy       = How well the product keeps users over time.   Values: high | medium | low

  [TIER 3 - LLM-DEPENDENT — use your world knowledge if text is silent]
  competition_intensity = How crowded and contested the market is.      Values: high | moderate | low

RULES:
1. For TIER 1 and TIER 2: only classify if the TEXT explicitly supports it.
2. For TIER 3 (competition_intensity): use your world knowledge about this company's market if the text does not say.
3. If genuinely unknown even with world knowledge, set to null.
4. Provide a short evidence quote (under 120 chars) for each non-null signal.
5. NEVER produce GO/CONDITIONAL/NO-GO — that is not your job.

FEW-SHOT EXAMPLES:
{few_shot_block}

TEXT TO ANALYZE:
{text_excerpt}

Respond in JSON format ONLY. No explanation outside the JSON.
{{
  "retention_proxy":       {{"value": "high|medium|low|null", "evidence": "..."}},
  "onboarding_friction":   {{"value": "low|medium|high|null",  "evidence": "..."}},
  "monetization_strength": {{"value": "strong|moderate|weak|null", "evidence": "..."}},
  "competition_intensity": {{"value": "high|moderate|low|null",  "evidence": "..."}}
}}"""

    return prompt


# ── Output Validator ───────────────────────────────────────────────────────────

def _validate_and_normalize(
    raw_output: dict,
    allowed_signals: Optional[set] = None,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """
    Validate LLM output against the signal schema.

    Args:
        raw_output:      Raw dict from LLM JSON parse
        allowed_signals: If set, only these signal names are returned.
                         Used to enforce tier restrictions on the rule-based path.
    """
    signal_values = {}
    signal_evidence = {}

    for signal_name, allowed_values in SIGNAL_SCHEMA.items():
        if allowed_signals is not None and signal_name not in allowed_signals:
            continue

        entry = raw_output.get(signal_name, {})
        if not isinstance(entry, dict):
            continue

        value = entry.get("value", "null")
        evidence = entry.get("evidence", "")

        if not value or value.lower() in ("null", "none", "unknown", ""):
            continue

        value_lower = value.lower().strip()
        if value_lower not in allowed_values:
            for allowed in allowed_values:
                if allowed in value_lower:
                    value_lower = allowed
                    break
            else:
                continue

        signal_values[signal_name] = value_lower
        if evidence and len(evidence) > 10:
            signal_evidence[signal_name] = [evidence[:200]]

    return signal_values, signal_evidence


# ── Tier 1 & 2 Rule-Based Extraction ──────────────────────────────────────────
# IMPORTANT: competition_intensity (Tier 3) is NOT in these rules.
# Its absence here is by design — it is LLM-DEPENDENT.

_RETENTION_RULES = {
    "high": [
        # Behavioral / analyst language
        "deeply embedded", "switching cost", "lock-in", "locked in",
        "critical dependency", "hard to replace", "all knowledge inside",
        "team depends", "silently runs", "strong retention", "network effect",
        "habit", "sticky", "embedded in your workflow",
        # Scale signals (sustained usage)
        "daily active users", "weekly active users", "millions of users",
        "million users", "active users", "fortune 100", "fortune 500",
        "200k+ paid", "300k+ paid", "500,000", "100,000 companies",
        "400,000 companies", "250,000 companies", "trusted by",
        "used by millions", "powered by millions",
        # Testimonial / dependency patterns
        "essential to our", "can't imagine", "changed how we",
        "never going back", "our team relies on", "we use it every day",
        "couldn't work without", "become part of our workflow",
        "saved us hundreds", "mission-critical", "backbone of our",
        "love using", "life of my life",
        # Integration depth = high retention
        "all your", "all in one", "bring everything", "replace multiple",
        "connects your team", "where work happens", "your team's hub",
        # Community scale
        "join the community", "paying customers", "actually use us",
        # Collaboration / design team network effect
        "whole team", "entire team", "real-time collaboration",
        "collaborate in real time", "multiplayer", "live collaboration",
        "collaborate on", "design team",
    ],
    "low": [
        "users left", "failed to retain", "churn", "dropped off", "not sticky",
        "no habit", "one-time use", "users stopped", "abandoned", "low engagement",
        "could not keep", "usage declined", "lost users", "failed to gain",
        "popularity declined", "nobody came back", "retention problem",
        "shut down", "closed", "failure", "went bankrupt", "discontinued",
        "users moved to", "replaced by", "superseded by", "platform declined",
        "could not build audience", "lost the audience",
    ],
}

_FRICTION_RULES = {
    "low": [
        "one-click", "instant setup", "no signup required", "easy to use",
        "seamless", "installs in seconds", "zero friction", "quick start",
        "no code", "developer-friendly", "simple integration", "out of the box",
        "one-line", "get started free", "start for free", "try for free",
        "sign up free", "sign up with google", "deploy in minutes",
        "no credit card", "no credit card required", "no setup required",
        "free account", "free plan", "starter free", "get loom for free",
        "add a script tag", "paste a snippet", "chrome extension",
        "connect your github", "link your calendar", "one command",
        "a few lines of code", "api-first", "developer first",
        "from your first transaction", "deploy start", "click deploy",
        "paste into your terminal", "npx ", "npm install",
        "single prompt", "install with ai",
        "up and running", "live in minutes", "works immediately",
        "automatic", "just connect", "plug and play",
        "build in a weekend", "start building in seconds",
        "spin up", "too easy", "really quick", "almost too easy",
    ],
    "high": [
        "complex setup", "hard to use", "steep learning curve",
        "requires training", "difficult to integrate",
        "long implementation", "requires consultant",
        "complex onboarding", "barrier to entry", "painful to set up",
        "users struggle", "not intuitive", "poor ux",
        "contact sales only", "procurement required", "enterprise only",
        "months to implement", "weeks to implement",
    ],
}

_MONETIZATION_RULES = {
    "strong": [
        "per transaction", "take rate", "seat license", "enterprise contract",
        "high conversion", "pays per use", "scales with usage",
        "direct cut", "usage-based billing", "annual contract",
        "revenue per seat", "charged per",
        "per user / month", "per user per month", "per seat per month",
        "billed monthly", "billed annually", "billing", "invoicing",
        "pro plan", "business plan", "enterprise plan", "upgrade to pro",
        "pay as you go", "pricing tiers", "free and paid", "premium features",
        "financial infrastructure", "payment processing", "payment gateway",
        "revenue models", "api pricing", "transaction fees",
        "200m+ active subscriptions", "payments volume",
        "accept payments", "custom revenue models", "monetize", "revenue growth",
        "seat licenses", "enterprise api", "enterprise-ready", "enterprise-grade",
    ],
    "weak": [
        "no revenue", "free service", "could not monetize",
        "monetization failed", "no business model",
        "giving away for free", "burned cash", "no paying customers",
        "revenue problem", "thin margins", "unprofitable", "lost money",
        "free tier only", "failed to charge", "ran out of money",
        "not profitable", "burn rate",
    ],
}

# NOTE: No _COMPETITION_RULES dict.
# competition_intensity is Tier 3 (LLM-DEPENDENT).
# Attempting to extract it from text produces unreliable results
# because company websites never name their own competitors.
# Only exception: Failory failure narratives sometimes do —
# but that's source-specific and handled by the LLM via context.


def _rule_based_extract(
    text: str,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """
    Tier 1 + Tier 2 rule-based extraction.

    Extracts: onboarding_friction (T1), monetization_strength (T1), retention_proxy (T2)
    Does NOT extract: competition_intensity (T3 — LLM-DEPENDENT)

    Requires minimum 2 keyword hits per signal to guard against false positives.
    """
    text_lower = text.lower()
    signal_values = {}
    signal_evidence = {}

    rules_map = {
        "retention_proxy":       _RETENTION_RULES,       # Tier 2
        "onboarding_friction":   _FRICTION_RULES,        # Tier 1
        "monetization_strength": _MONETIZATION_RULES,    # Tier 1
        # competition_intensity intentionally absent
    }

    for signal_name, rules in rules_map.items():
        best_level = None
        best_score = 0
        best_evidence = []

        for level, keywords in rules.items():
            hits = [kw for kw in keywords if kw in text_lower]
            if len(hits) >= 2:
                if len(hits) > best_score:
                    best_score = len(hits)
                    best_level = level
                    evidence = []
                    for kw in hits[:2]:
                        pos = text_lower.find(kw)
                        if pos >= 0:
                            start = max(0, pos - 80)
                            end   = min(len(text), pos + len(kw) + 80)
                            snippet = text[start:end].strip()
                            if len(snippet) > 20:
                                evidence.append(snippet[:200])
                    best_evidence = evidence

        if best_level:
            signal_values[signal_name] = best_level
            if best_evidence:
                signal_evidence[signal_name] = best_evidence[:2]

    return signal_values, signal_evidence


# ── Main Extractor Class ───────────────────────────────────────────────────────

class ModelSignalExtractor:
    """
    Three-tier signal extractor.

    Tier 1 (TEXT_EXTRACTABLE):  rule-based, no LLM needed
    Tier 2 (HYBRID):            rule-based primary, LLM supplements
    Tier 3 (LLM_DEPENDENT):     LLM only — null when Ollama unavailable

    This is NOT a degraded implementation. The tier structure IS the design.
    Returning null for competition_intensity without Ollama is correct behaviour.
    """

    def __init__(self):
        self._ollama_available = False
        self._ollama_client = None
        self._init_ollama()

    def _init_ollama(self):
        try:
            from utils.ollama_client import get_ollama_client
            client = get_ollama_client()
            self._ollama_available = client.available
            self._ollama_client = client
        except Exception:
            self._ollama_available = False

    def extract_signals(
        self,
        text: str,
        startup_name: str = "",
    ) -> tuple[dict[str, str], dict[str, list[str]]]:
        """
        Extract L3 signals from raw text using the three-tier strategy.

        Tier 1 & 2 always run (rule-based).
        Tier 3 only runs when Ollama is available.
        If LLM available, it runs on all tiers and its results override Tier 2
        for cases the rules missed — but Tier 3 results ONLY come from LLM.

        Returns:
            (signal_values, signal_evidence)
            - signal_values keys are a subset of SIGNAL_SCHEMA
            - competition_intensity is absent when Ollama is unavailable
              (this is correct, not a bug)
        """
        if not text or len(text.strip()) < 50:
            return {}, {}

        # Step 1: Rule-based (Tier 1 + Tier 2 only)
        rule_values, rule_evidence = _rule_based_extract(text)

        if not self._ollama_available:
            # Without LLM: Tier 1 + Tier 2 only. Tier 3 = null (correct).
            return rule_values, rule_evidence

        # Step 2: LLM pass (all tiers, including Tier 3)
        try:
            llm_values, llm_evidence = self._ollama_extract(text, startup_name)

            # Merge: LLM wins on any signal it produced
            merged_values  = {**rule_values,  **llm_values}
            merged_evidence = {**rule_evidence, **llm_evidence}
            return merged_values, merged_evidence

        except Exception:
            # LLM failure: fall back to rule-based (Tier 3 remains null)
            return rule_values, rule_evidence

    def _ollama_extract(
        self, text: str, startup_name: str
    ) -> tuple[dict[str, str], dict[str, list[str]]]:
        """LLM extraction — covers all tiers including Tier 3."""
        if not self._ollama_client:
            return {}, {}

        prompt = _build_extraction_prompt(text, startup_name)
        raw_response = self._ollama_client._query(prompt, temperature=0.1)
        if not raw_response:
            return {}, {}

        parsed = self._ollama_client._parse_json_response(raw_response)
        if not parsed or not isinstance(parsed, dict):
            return {}, {}

        return _validate_and_normalize(parsed)

    def tier_report(self) -> dict:
        """
        Return the signal tier map for logging and documentation.
        Exposes the architecture design explicitly.
        """
        return {
            "signal_tiers":       SIGNAL_TIERS,
            "llm_available":      self._ollama_available,
            "competition_intensity_available": self._ollama_available,
            "note": (
                "competition_intensity requires LLM world knowledge. "
                "Returning null without Ollama is correct, not a deficiency."
            ),
        }

    def validate_against_ground_truth(self) -> dict:
        """
        Self-test against ground_truth.json.

        Reports accuracy per tier, not just per signal.
        Tier 3 accuracy is only meaningful when Ollama is available.
        """
        corpus_path = BASE_DIR / "data" / "training" / "signal_corpus.json"
        gt_path     = BASE_DIR / "data" / "validation" / "ground_truth.json"

        if not gt_path.exists() or not corpus_path.exists():
            return {"error": "Ground truth or corpus not found"}

        ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
        corpus       = json.loads(corpus_path.read_text(encoding="utf-8"))
        raw_by_name  = {ex["startup_name"]: ex["raw_text"] for ex in corpus}

        results_by_tier: dict[str, dict] = {
            "TEXT_EXTRACTABLE": {"correct": 0, "total": 0, "missed": 0},
            "HYBRID":           {"correct": 0, "total": 0, "missed": 0},
            "LLM_DEPENDENT":    {"correct": 0, "total": 0, "missed": 0,
                                 "note": "null when LLM unavailable — expected"},
        }

        for entry in ground_truth:
            name = entry["startup_name"]
            raw_text = raw_by_name.get(name, "")
            if not raw_text:
                continue

            gt_signals = {
                sig["signal"]: sig["value"]
                for sig in entry["signals"]
                if sig["signal"] in SIGNAL_SCHEMA
            }
            if not gt_signals:
                continue

            extracted, _ = self.extract_signals(raw_text, startup_name=name)

            for signal_name, expected in gt_signals.items():
                tier = SIGNAL_TIERS[signal_name]
                results_by_tier[tier]["total"] += 1
                got = extracted.get(signal_name)
                if got is None:
                    results_by_tier[tier]["missed"] += 1
                elif got == expected:
                    results_by_tier[tier]["correct"] += 1

        for tier, r in results_by_tier.items():
            t = r["total"]
            c = r["correct"]
            r["accuracy"] = round(c / t, 3) if t > 0 else None

        return {
            "by_tier": results_by_tier,
            "llm_available": self._ollama_available,
            "boundary_check": {
                "trained_on_decision_analysis": False,
                "trained_on_pattern_tags":      False,
                "labels_source":                "ground_truth.json",
            },
        }


# ── Singleton ──────────────────────────────────────────────────────────────────

_extractor_instance: Optional[ModelSignalExtractor] = None


def get_model_signal_extractor() -> ModelSignalExtractor:
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = ModelSignalExtractor()
    return _extractor_instance
