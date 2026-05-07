"""
midan.l0_gate — input validation gate.

Strict, blocking checks applied before any pipeline analysis. Nine
deterministic checks plus an LLM borderline arbiter. Every rejection is
blocking; there is no "advisory pass-through" path.
"""
from midan.core import *  # noqa: F401,F403


# ── extracted from api.py ─────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════
# LAYER 0 — IDEA SANITY CHECKER  (full redesign)
#
# Architecture:
#   Input idea
#      │
#      ├─ [D1] Length gate          < 5 words → too_short             (conf 0.99)
#      ├─ [D2] Logical impossibility  physics/logic violation          (conf 0.97)
#      ├─ [D3] No revenue model       free-money / charity / donation  (conf 0.96)
#      ├─ [D4] No value exchange       free-everything, no B2B path    (conf 0.92)
#      ├─ [D5] Unsustainable economics pay-users / pyramid / inf-burn  (conf 0.93)
#      ├─ [D6] Vague / non-actionable  no sector+customer+problem      (conf 0.88)
#      └─ [LLM] Arbiter               structured JSON verdict          (conf 0.90)
#
# On rejection: returns {valid:False, rejection_type, logical_validity_score,
#               rejection_confidence, message, one_line_verdict, what_is_missing}
# On pass:      returns {valid:True, logical_validity_score}
#
# logical_validity_score (0–1):
#   1.0 = definitively a real business
#   0.0 = definitively not a business
#   Separate from market_confidence (SVM) — only valid ideas get market analysis.
# ═══════════════════════════════════════════════════════════════

import logging as _logging
_L0_LOG = _logging.getLogger("midan.layer0")

# ── D2: Logical / physical impossibilities ─────────────────────────────────────
_L0_IMPOSSIBLE = [
    'make people live forever', 'eliminate aging completely',
    'cure all diseases instantly', 'end all disease',
    'eliminate all poverty overnight', 'end world hunger instantly',
    'predict the future with certainty', 'predict lottery numbers',
    'time travel service', 'time travel startup', 'build a time machine',
    'teleportation startup', 'teleportation service',
    'perpetual motion machine', 'free energy machine',
    'guaranteed 100% accuracy', 'zero error rate guaranteed',
    'never fail', 'impossible to lose money', '100% guaranteed profit',
]

# ── D3: No revenue model — free-money, charity, pure redistribution ────────────
# These describe moving money to people with no charging mechanism on top.
_L0_FREE_MONEY = [
    'give people money', 'giving people money', 'give money for free',
    'giving money for free', 'giving away money', 'give away money',
    'pay people for nothing', 'pay people to do nothing',
    'pay everyone', 'give everyone money', 'give money to everyone',
    'distribute money', 'distribute free cash', 'give cash to everyone',
    'giving cash to people', 'free money for everyone', 'free money to people',
    'money for free', 'give users money', 'paying users to exist',
    'ubi startup', 'universal basic income app',
    'give people free money', 'hand out money',
]

# ── D4: No value exchange — free-for-all with zero monetization path ───────────
# Triggers when ALL of: free + no B2B/subscription/commission/enterprise signal
_L0_FREE_EVERYTHING = [
    'completely free for everyone', 'totally free forever',
    'free for all users', 'free to everyone forever',
    '100% free no cost', 'always free no charge',
    'free forever with no premium', 'no cost ever',
    'free with no monetization', 'free app no revenue',
    'no business model just free', 'free service forever',
]
_L0_MONETIZATION_RESCUE = [
    'subscription', 'enterprise', 'b2b', 'premium', 'freemium',
    'commission', 'saas', 'paid plan', 'license', 'revenue', 'monetize',
    'charge', 'fee', 'invoice', 'per user', 'per seat', 'api key',
    'white label', 'data', 'ads', 'advertising', 'sponsored',
]

# ── D5: Unsustainable economics — pay-to-play / pyramid / infinite burn ────────
_L0_UNSUSTAINABLE = [
    'pay users to join', 'pay users to sign up', 'pay people to use',
    'we pay users', 'users get paid just for using',
    'earn by doing nothing', 'earn money for free',
    'infinite returns', 'guaranteed returns', 'guaranteed profits',
    'pyramid', 'ponzi', 'mlm model', 'multi-level marketing',
    'recruit people to earn', 'earn by recruiting',
    'fund it ourselves forever', 'no need to make money',
    'investors keep funding forever', 'never need revenue',
    'we absorb all costs forever', 'charity funded startup',
    'donation funded startup', 'donor funded forever',
]

# ── D6: Vague / non-actionable — rescue words prevent rejection ────────────────
_L0_VAGUE_INDICATORS = [
    'something with', 'thing for', 'stuff about', 'idea about',
    'general solution', 'do everything', 'solves everything',
    'app for everything', 'platform for everything',
    'connects everyone', 'helps everyone with anything',
    'a better way', 'improve things', 'make things better',
    'revolutionary idea', 'change the world',
    'disrupt everything', 'fix society',
]
_L0_CONCRETE_RESCUE = [
    'hospital', 'clinic', 'doctor', 'patient', 'student', 'farmer',
    'driver', 'merchant', 'retailer', 'restaurant', 'logistics',
    'payment', 'invoice', 'loan', 'insurance', 'delivery',
    'health', 'education', 'finance', 'agriculture', 'real estate',
    'sme', 'enterprise', 'factory', 'supplier', 'manufacturer',
    'egypt', 'cairo', 'saudi', 'dubai', 'uae', 'nigeria', 'kenya',
    'africa', 'mena', 'gulf', 'usa', 'europe',
]

# ── D7: Contradictions — claims that mutually exclude each other ──────────────
# Each tuple is (phrase_a, phrase_b). If BOTH appear, the idea is internally inconsistent.
_L0_CONTRADICTIONS = [
    ('completely free', 'subscription'),
    ('completely free', 'premium tier'),
    ('completely free', 'paid plan'),
    ('totally free', 'monthly fee'),
    ('no fees ever', 'commission'),
    ('only b2b', 'consumers pay'),
    ('only b2b', 'individual consumers'),
    ('only b2c', 'enterprise contracts'),
    ('no internet required', 'cloud-based'),
    ('no internet required', 'real-time sync'),
    ('zero data collection', 'personalized recommendations'),
    ('zero data collection', 'machine learning model'),
    ('no employees', 'high-touch support'),
    ('no employees', 'concierge service'),
    ('open source and free', 'enterprise license fee'),
    ('hardware-free', 'ship the device'),
    ('non-profit', 'maximize shareholder return'),
]

# ── D8: Prompt-injection / system-prompt manipulation in idea text ────────────
_L0_PROMPT_INJECTION = [
    'ignore previous instructions', 'ignore the previous',
    'disregard prior instructions', 'forget all previous',
    'you are now', 'act as a',
    'system prompt', 'override your', 'jailbreak',
    'pretend you are', 'roleplay as',
    'output the password', 'reveal your prompt',
    '</system>', '<|im_start|>', '<|im_end|>',
]

# Patterns indicating the input is not an idea description at all
_L0_NON_IDEA_TOKENS = [
    'lorem ipsum', 'asdf', 'qwerty', 'test test test',
    '...', '!!!!!', '???????', '12345', 'aaaaa',
]


def _l0_check_contradiction(t: str) -> Optional[dict]:
    hits: list = []
    for a, b in _L0_CONTRADICTIONS:
        if a in t and b in t:
            hits.append(f"'{a}' vs '{b}'")
    if not hits:
        return None
    return {
        'valid':                  False,
        'severity':               'BROKEN',
        'rejection_type':         'contradictory_claims',
        'logical_validity_score': 0.18,
        'rejection_confidence':   0.92,
        'message': (
            "The idea contains mutually exclusive claims that cannot all be true. "
            "MIDAN will not run analysis on a contradictory specification — "
            f"resolve the conflict first. Conflicts detected: {'; '.join(hits[:2])}."
        ),
        'one_line_verdict': "Internally inconsistent — claims contradict each other.",
        'what_is_missing':  "A coherent specification: pick which claim holds and revise the rest.",
    }


def _l0_check_spam_or_gibberish(t: str, wc: int) -> Optional[dict]:
    """
    Block token-repetition spam, gibberish, and obvious non-idea inputs.
    Heuristic: if the most common token covers > 40% of tokens AND wc ≥ 8,
    or if the alpha-token ratio is < 0.5 (mostly punctuation/digits),
    or if a known non-idea pattern is present.
    """
    for p in _L0_NON_IDEA_TOKENS:
        if p in t:
            return {
                'valid':                  False,
                'severity':               'BROKEN',
                'rejection_type':         'spam_or_gibberish',
                'logical_validity_score': 0.05,
                'rejection_confidence':   0.95,
                'message': (
                    "Input is not a business idea — appears to be placeholder text, "
                    "test characters, or non-idea content. MIDAN evaluates real concepts only."
                ),
                'one_line_verdict': "Not an idea — placeholder or non-content input.",
                'what_is_missing':  "An actual idea description: customer, problem, mechanism.",
            }

    tokens = [tok for tok in t.split() if tok]
    if len(tokens) >= 8:
        from collections import Counter as _Counter
        most_common, count = _Counter(tokens).most_common(1)[0]
        if count / len(tokens) > 0.40 and len(most_common) > 1:
            return {
                'valid':                  False,
                'severity':               'BROKEN',
                'rejection_type':         'spam_or_gibberish',
                'logical_validity_score': 0.06,
                'rejection_confidence':   0.93,
                'message': (
                    f"Input is dominated by repetition ('{most_common}' covers "
                    f"{int(count/len(tokens)*100)}% of tokens). This is not a "
                    "structured business description."
                ),
                'one_line_verdict': "Token-repetition spam, not a business idea.",
                'what_is_missing':  "A structured description with a customer, a problem, and a mechanism.",
            }

    if t and len(t) >= 10:
        alpha_chars = sum(1 for c in t if c.isalpha() or c.isspace())
        if alpha_chars / max(len(t), 1) < 0.55:
            return {
                'valid':                  False,
                'severity':               'BROKEN',
                'rejection_type':         'spam_or_gibberish',
                'logical_validity_score': 0.08,
                'rejection_confidence':   0.90,
                'message': (
                    "Input is mostly non-alphabetic — punctuation, digits, or symbols "
                    "dominate. This is not a readable business description."
                ),
                'one_line_verdict': "Unreadable — non-alphabetic content dominates.",
                'what_is_missing':  "Plain-language description of the customer, problem, and solution.",
            }
    return None


def _l0_check_prompt_injection(t: str) -> Optional[dict]:
    for p in _L0_PROMPT_INJECTION:
        if p in t:
            return {
                'valid':                  False,
                'severity':               'BROKEN',
                'rejection_type':         'adversarial_prompt',
                'logical_validity_score': 0.02,
                'rejection_confidence':   0.97,
                'message': (
                    "Input contains prompt-manipulation patterns rather than a business "
                    "description. MIDAN treats this as adversarial input and will not "
                    "process it as an idea."
                ),
                'one_line_verdict': "Adversarial input — prompt manipulation, not an idea.",
                'what_is_missing':  "A real business description: customer, problem, solution mechanism.",
            }
    return None


# ── Per-dimension check functions ──────────────────────────────────────────────

def _l0_check_length(t: str, wc: int) -> Optional[dict]:
    if wc >= 5:
        return None
    return {
        'valid':                  False,
        'severity':               'INCOMPLETE',
        'rejection_type':         'too_short',
        'logical_validity_score': 0.20,
        'rejection_confidence':   0.97,
        'message': (
            "Too short to evaluate. A startup idea needs at minimum: "
            "the problem, who it affects, and how your solution works."
        ),
        'one_line_verdict': "Not enough information to evaluate as a business concept.",
        'what_is_missing':  "Problem description, target customer, and solution mechanism.",
    }


def _l0_check_impossibility(t: str) -> Optional[dict]:
    for p in _L0_IMPOSSIBLE:
        if p in t:
            return {
                'valid':                  False,
                'severity':               'IMPOSSIBLE',
                'rejection_type':         'logical_impossibility',
                'logical_validity_score': 0.02,
                'rejection_confidence':   0.97,
                'message': (
                    "This describes a physical or logical impossibility — "
                    "a constraint no commercial model can overcome. "
                    "MIDAN evaluates ideas buildable with current technology "
                    "and a viable economic structure."
                ),
                'one_line_verdict': "Physically or logically impossible — no commercial path exists.",
                'what_is_missing':  "A constraint that technology or markets can actually solve.",
            }
    return None


def _l0_check_no_revenue_model(t: str) -> Optional[dict]:
    for p in _L0_FREE_MONEY:
        if p in t:
            return {
                'valid':                  False,
                'severity':               'IMPOSSIBLE',
                'rejection_type':         'no_revenue_model',
                'logical_validity_score': 0.04,
                'rejection_confidence':   0.96,
                'message': (
                    "This describes redistributing money to people, not a business. "
                    "A startup requires a revenue mechanism where the value delivered "
                    "to customers generates income that exceeds the cost of delivery. "
                    "Giving away money violates that constraint — it is a subsidy, "
                    "charity, or government program, not a commercial entity. "
                    "Who pays, why, and how much?"
                ),
                'one_line_verdict': "No revenue mechanism — value flows out with nothing returning.",
                'what_is_missing':  "A paying customer, a revenue model, and a unit economics path to profitability.",
            }
    return None


def _l0_check_no_value_exchange(t: str) -> Optional[dict]:
    matched_free = any(p in t for p in _L0_FREE_EVERYTHING)
    if not matched_free:
        return None
    has_monetization = any(m in t for m in _L0_MONETIZATION_RESCUE)
    if has_monetization:
        return None
    return {
        'valid':                  False,
        'severity':               'BROKEN',
        'rejection_type':         'no_value_exchange',
        'logical_validity_score': 0.12,
        'rejection_confidence':   0.90,
        'message': (
            "This describes a free-for-all service with no described monetization path. "
            "A viable business requires someone to pay for the value created — "
            "either directly (subscription, commission, license) or indirectly (B2B, data, ads). "
            "Offering everything free forever with no revenue source is not a startup, "
            "it is a cost center. Add a paying segment."
        ),
        'one_line_verdict': "Free service with no monetization path described — needs a paying segment.",
        'what_is_missing':  "A paying segment: B2B, premium tier, commission, subscription, or data monetization.",
    }


def _l0_check_unsustainable_economics(t: str) -> Optional[dict]:
    for p in _L0_UNSUSTAINABLE:
        if p in t:
            return {
                'valid':                  False,
                'severity':               'IMPOSSIBLE',
                'rejection_type':         'unsustainable_economics',
                'logical_validity_score': 0.05,
                'rejection_confidence':   0.93,
                'message': (
                    "This describes an economically unsustainable model — "
                    "either a pyramid structure, a model that pays users with no upstream revenue, "
                    "or one that explicitly avoids needing to generate income. "
                    "Every viable startup must have a path where revenue ≥ cost at some scale. "
                    "This model has no such path."
                ),
                'one_line_verdict': "Structurally unsustainable — the economics break by design.",
                'what_is_missing':  "A unit economics model where revenue at scale exceeds cost of delivery.",
            }
    return None


def _l0_check_vague(t: str, wc: int) -> Optional[dict]:
    if wc > 20:
        return None
    matched_vague = any(p in t for p in _L0_VAGUE_INDICATORS)
    if not matched_vague:
        return None
    has_concrete = any(c in t for c in _L0_CONCRETE_RESCUE)
    if has_concrete:
        return None
    # INCOMPLETE is now BLOCKING — vague ideas must be clarified before analysis runs.
    # Previous behavior (valid=True pass-through) let weak inputs contaminate L1–L4.
    return {
        'valid':                  False,
        'severity':               'INCOMPLETE',
        'rejection_type':         'vague_non_actionable',
        'logical_validity_score': 0.35,
        'rejection_confidence':   0.82,
        'message': (
            "This idea is too vague to evaluate. MIDAN will not run analysis on "
            "an under-defined concept — narrow to one customer, one problem, one "
            "mechanism, then resubmit."
        ),
        'one_line_verdict': "Idea is under-defined — clarification required before analysis.",
        'what_is_missing':  "A specific problem, a specific customer segment, and a concrete solution mechanism.",
    }


def _l0_llm_arbiter(idea_text: str) -> Optional[dict]:
    """
    Structured LLM classification for borderline cases.
    Returns a rejection dict if the LLM identifies a failure, None if it passes.
    Only called after all pattern checks pass.
    """
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not (GROQ_CLIENT and groq_key and groq_key != "dummy"):
        return None

    prompt = dedent(f"""
        You are a startup viability analyst. Evaluate whether this text describes a
        legitimate business concept that could realistically generate revenue.

        Text: "{idea_text}"

        A business is INVALID if ANY of these are true:
        - No revenue mechanism (no one pays, pure charity/donation/redistribution)
        - No value exchange (gives things away with no monetization path)
        - Structurally unsustainable economics (pays users with no upstream revenue, pyramid)
        - Logically or physically impossible to execute
        - So vague it describes no specific product, customer, or problem

        A business is VALID even if it is early-stage, unproven, competitive, or risky —
        as long as it has a plausible path to revenue from a real customer.

        Respond ONLY with this exact JSON (no markdown, no explanation):
        {{
          "is_valid": true,
          "primary_failure": null,
          "rejection_confidence": 0.0,
          "one_line_verdict": "Brief verdict in one sentence.",
          "what_is_missing": "What would make it valid, or null if already valid."
        }}

        primary_failure must be exactly one of:
        "no_revenue_model" | "no_value_exchange" | "unsustainable_economics" |
        "logical_impossibility" | "vague_non_actionable" | null
    """).strip()

    try:
        resp = GROQ_CLIENT.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            temperature=0.0,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content.strip())
        if data.get("is_valid", True):
            return None

        rtype = data.get("primary_failure") or "not_a_business_idea"
        conf  = float(data.get("rejection_confidence", 0.88))
        conf  = max(0.85, min(0.96, conf))

        _severity_map = {
            "no_revenue_model":       "IMPOSSIBLE",
            "unsustainable_economics":"IMPOSSIBLE",
            "logical_impossibility":  "IMPOSSIBLE",
            "no_value_exchange":      "BROKEN",
            "vague_non_actionable":   "INCOMPLETE",
        }
        _reason_messages = {
            "no_revenue_model":       "No revenue mechanism identified — no paying customer, no charging model.",
            "no_value_exchange":      "No value exchange: the idea gives value away with no monetization path.",
            "unsustainable_economics":"The economics are structurally broken — costs exceed revenue by design.",
            "logical_impossibility":  "This is physically or logically impossible to execute commercially.",
            "vague_non_actionable":   "Too vague to evaluate — no specific problem, customer, or mechanism stated.",
        }
        return {
            'valid':                  False,
            'severity':               _severity_map.get(rtype, 'BROKEN'),
            'rejection_type':         rtype,
            'logical_validity_score': round(1.0 - conf, 2),
            'rejection_confidence':   conf,
            'message':                _reason_messages.get(rtype, "This does not describe a viable business."),
            'one_line_verdict':       str(data.get("one_line_verdict", "Not a viable business concept.")),
            'what_is_missing':        str(data.get("what_is_missing") or "A revenue model and a paying customer."),
        }
    except Exception as llm_err:
        _L0_LOG.warning(f"[L0] LLM arbiter failed: {llm_err!r} — passing idea through")
        return None


# ── Rejection pattern store (session-level, last 200 rejections) ──────────────
import time as _time

_REJECTION_LOG: list = []


def _log_rejection(rejection_type: str, severity: str, idea_snippet: str) -> None:
    _REJECTION_LOG.append({
        'ts':       _time.time(),
        'type':     rejection_type,
        'severity': severity,
        'snippet':  idea_snippet[:80],
    })
    if len(_REJECTION_LOG) > 200:
        _REJECTION_LOG.pop(0)


# ── Rule-based how-to-fix fallbacks (specific > generic) ──────────────────────
_L0_FIX_FALLBACKS: Dict[str, list] = {
    'logical_impossibility': [
        "Reframe the promise to something achievable: instead of eliminating the problem entirely, offer a measurable reduction (e.g., 60% cost cut in 30 days) — that is a product, not a fantasy.",
        "Find the addressable subset: which part of this problem CAN be solved commercially with current technology and a viable cost structure? Start there.",
    ],
    'no_revenue_model': [
        "Identify who benefits commercially from this value existing — that entity (employer, hospital, government, platform) is your real paying customer, not the end user who receives the money.",
        "Flip to B2B: if consumers cannot pay, which business gains economic benefit when consumers have this service? Charge that business.",
        "Structure a fee-on-transaction model: if value is being moved, take a cut of the movement — wage advances, micro-insurance, and credit scoring all do this.",
    ],
    'no_value_exchange': [
        "Define your paying segment: B2B (charge companies that benefit), premium tier (charge power users), or API/data (charge developers who build on top).",
        "Replace 'completely free' with freemium: free for basic access, paid for volume, integrations, analytics, or white-label. Remove 'forever' from your pitch.",
    ],
    'unsustainable_economics': [
        "Replace the earn-by-recruiting structure with genuine product value — users must stay because the product solves a problem, not because of financial incentives that require more users to fund.",
        "Identify what is upstream of the users getting paid: which business generates the revenue that would fund those payments? That entity is your customer.",
    ],
    'vague_non_actionable': [
        "Compress to one sentence: 'We help [specific customer type] [do specific thing] by [specific mechanism].' That sentence is your minimum viable description.",
        "Name your first customer: a real person, real role, real company type — then describe the specific workflow they would change if your product existed.",
    ],
    'too_short': [
        "Add three things in one message: the customer type, the pain they have, and how you solve it.",
        "Include where (country/market), who (B2B or B2C, which segment), and what specifically (app, platform, service, tool).",
    ],
    'not_a_business_idea': [
        "Define what value you create, who receives it, and who pays for it — all three must be answered for a business to exist.",
        "Describe the specific problem your first 10 customers have and why they would pay to solve it over a free alternative.",
    ],
    'contradictory_claims': [
        "Pick the anchor: decide which claim is non-negotiable (e.g., 'fully free') and revise the others to be consistent with it.",
        "If you want both halves of the contradiction, restructure as tiers — free for one segment, paid for another — and describe each segment separately.",
    ],
    'spam_or_gibberish': [
        "Submit an actual idea description — one or two sentences naming the customer, the problem, and the mechanism.",
        "Replace placeholder/test text with a real concept: 'We help [customer] [do X] by [mechanism].'",
    ],
    'adversarial_prompt': [
        "Submit a business idea, not instructions to the system. MIDAN evaluates concepts, it does not take role-play directives.",
        "Describe the customer, the problem, and how you intend to solve it — that is the only input MIDAN processes.",
    ],
    'l1_insufficient_confidence': [
        "Add the missing component(s) explicitly: business model (subscription/marketplace/SaaS/commission/service), target segment (B2B/B2C/B2G), and current stage (idea/validation/MVP/growth).",
        "State the revenue mechanism in one phrase: who pays you, how often, and for what — that one sentence resolves most ambiguity.",
    ],
    'l1_inconsistent_schema': [
        "The combination of sector, business model, and segment you described is internally inconsistent. Pick the anchor (sector or model) and align the others.",
        "Restate the idea as a single sentence: '[Customer] in [market] pays [mechanism] for [outcome]' — that surfaces the actual model.",
    ],
}


def _l0_how_to_fix(idea_text: str, rejection_type: str) -> list:
    """
    Generate 2–3 specific, idea-grounded fix suggestions.
    LLM first (grounded in the actual idea text); rule-based fallback.
    """
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if GROQ_CLIENT and groq_key and groq_key != "dummy":
        try:
            prompt = dedent(f"""
                A startup idea was rejected because of: {rejection_type.replace('_', ' ')}

                The idea: "{idea_text}"

                Give EXACTLY 2 or 3 specific ways to fix this idea to become a viable business.

                HARD RULES — violations are rejected:
                - Each fix MUST reference the actual idea, not give generic advice
                - Be concrete: name a mechanism, model, or specific approach
                - Reference the specific market/sector if detectable from the idea
                - NOT: "Add a revenue model" (too generic)
                - NOT: "Consider monetization" (too generic)
                - YES: "In Egypt, employer-backed wage advance models work because employers pay the fee as an HR benefit — target payroll managers, not the workers receiving advances"
                - YES: "Charge hospitals a per-referral fee for triaging patients before they arrive — the hospital saves ER costs, not the patients who pay nothing"

                Return ONLY valid JSON:
                {{"fixes": ["Fix 1 specific to this idea", "Fix 2 specific to this idea"]}}
            """).strip()

            resp = GROQ_CLIENT.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=GROQ_MODEL,
                temperature=0.25,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            data  = json.loads(resp.choices[0].message.content.strip())
            fixes = data.get('fixes', [])
            if isinstance(fixes, list) and len(fixes) >= 2:
                return [str(f) for f in fixes[:3]]
        except Exception as e:
            _L0_LOG.warning(f"[L0] how_to_fix LLM failed: {e!r}")

    return _L0_FIX_FALLBACKS.get(rejection_type, _L0_FIX_FALLBACKS['not_a_business_idea'])

def _layer0_sanity_check(idea_text: str) -> dict:
    """
    Layer 0 orchestrator — strict, blocking gate. Runs deterministic checks
    then the LLM arbiter. EVERY rejection is blocking — there is no
    "INCOMPLETE pass-through" path. Weak/contradictory/adversarial inputs
    must not reach L1–L4.

    Priority order (fast → slow, high-confidence → low-confidence):
      D1 Length → D2 Impossibility → D3 No Revenue → D4 No Value Exchange
      → D5 Unsustainable Econ → D7 Contradiction → D8 Spam/Gibberish
      → D9 Prompt Injection → D6 Vague → LLM Arbiter

    Returns:
      {'valid': True,  'logical_validity_score': float}     — proceed
      {'valid': False, 'severity': ..., 'rejection_type': ...,
       'rejection_confidence': float, 'message': str,
       'one_line_verdict': str, 'what_is_missing': str, 'how_to_fix': list}
    """
    t  = idea_text.lower().strip()
    wc = len(t.split())

    checks = [
        _l0_check_length(t, wc),
        _l0_check_impossibility(t),
        _l0_check_no_revenue_model(t),
        _l0_check_no_value_exchange(t),
        _l0_check_unsustainable_economics(t),
        _l0_check_contradiction(t),
        _l0_check_spam_or_gibberish(t, wc),
        _l0_check_prompt_injection(t),
        _l0_check_vague(t, wc),
    ]
    for result in checks:
        if result is None:
            continue
        # All checks are now blocking — no valid=True early return.
        result['how_to_fix'] = _l0_how_to_fix(idea_text, result['rejection_type'])
        _L0_LOG.info(
            f"[L0] REJECTED severity={result.get('severity','?')} "
            f"type={result['rejection_type']} conf={result['rejection_confidence']:.0%} "
            f"idea='{idea_text[:60]}'"
        )
        _log_rejection(result['rejection_type'], result.get('severity', 'BROKEN'), idea_text)
        return result

    # All deterministic checks passed → LLM arbiter for borderline cases
    llm_result = _l0_llm_arbiter(idea_text)
    if llm_result is not None:
        llm_result['how_to_fix'] = _l0_how_to_fix(idea_text, llm_result['rejection_type'])
        _L0_LOG.info(
            f"[L0] LLM REJECTED severity={llm_result.get('severity','?')} "
            f"type={llm_result['rejection_type']} conf={llm_result['rejection_confidence']:.0%}"
        )
        _log_rejection(llm_result['rejection_type'], llm_result.get('severity', 'BROKEN'), idea_text)
        return llm_result

    _L0_LOG.debug(f"[L0] PASSED — idea='{idea_text[:60]}'")
    return {'valid': True, 'logical_validity_score': 0.92}





# Export everything defined in this module — including underscore-prefixed
# helpers — so other midan submodules can wildcard-import the full surface.
__all__ = [name for name in list(globals().keys()) if not name.startswith('__')]
