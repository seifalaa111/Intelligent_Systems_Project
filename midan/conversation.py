"""
midan.conversation — intent classification + post-decision routing.

Stateless conversation helpers (no session memory). Each turn is classified
into an intent (CASUAL / GREETING / META / PARTIAL_IDEA / ANALYSIS_REQUEST
/ OVERRIDE_COMMAND / CLARIFICATION) and post-decision turns are routed by
L4 decision_state into one of four behavioral modes (STANDARD_ADVISOR,
RESOLVING_CONFLICT, ADVISORY_ONLY, RE_CLARIFY).
"""
from midan.core import *  # noqa: F401,F403


# ── extracted from api.py ─────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════
# /interact — 5-gate intent state machine (pre-analysis routing)
# ═══════════════════════════════════════════════════════════════

_GREET_TOKENS = {
    'hi', 'hello', 'hey', 'yo', 'sup', 'hola', 'marhaba', 'ahlan',
    'good morning', 'good evening', 'good afternoon', 'howdy', 'greetings',
}
_META_PHRASES = [
    'what do you do', 'who are you', 'how does this work', 'what is midan',
    'what can you do', 'explain to me', 'help me understand', 'tell me about yourself',
    'what are you', 'how do i use this', 'what does midan',
]
_VAGUE_STARTERS = [
    'i have an idea', 'i have a startup', 'i want to share', 'i want to tell',
    'i would like to', 'i am thinking', 'i am building', 'i am working',
    'i am developing', 'i am creating', 'what do you think', 'give me feedback',
    'give me your thoughts', 'can you analyze', 'i need help with',
    'i have something', 'i have a concept', 'let me tell you',
    'so i was thinking', 'i was thinking', 'been working on',
    'working on something', 'building something', 'have an idea', 'had an idea',
    'i want to build', 'i want to create', 'i want to start',
]

# ── Casual / personal conversation — no idea content ──────────────────────────
_CASUAL_PREFIXES = [
    'call me ', 'my name is ', 'i am ', "i'm ", 'im ',
    'i live in ', 'i am from ', "i'm from ", 'from ',
    'i studied ', 'just checking', 'just wanted to say',
]
_CASUAL_SHORT_SET = {
    'ok', 'okay', 'cool', 'nice', 'great', 'got it', 'understood',
    'makes sense', 'i see', 'interesting', 'lol', 'haha', 'yes', 'no',
    'sure', 'right', 'alright', 'perfect', 'noted', 'agreed', 'good',
    'awesome', 'wow', 'really', 'oh', 'ah', 'hmm', 'k', 'yep', 'nope',
    'thanks again', 'ty', 'thx', 'not yet', 'maybe', 'later', 'soon',
}

# ── Override commands — user explicitly demands immediate analysis ─────────────
_OVERRIDE_COMMANDS = [
    'analyze now', 'run it', 'full breakdown', 'analyze this', 'just analyze',
    'go ahead and analyze', 'run analysis', 'skip the questions', 'stop asking',
    'just go', 'run it now', 'do the analysis', 'show me the analysis',
    'run the analysis', 'just run it', 'analyze it now', 'break it down',
    'full analysis', 'start analysis', 'do it now', 'just do it',
    'analyze already', 'just analyze it', 'analyze please', 'please analyze',
    'just run', 'run the model', 'go ahead', 'proceed', 'continue with analysis',
]
_PROBLEM_SIGNALS = [
    'problem','pain','issue','struggle','challenge','gap','need','lack',
    'inefficient','inefficiency','expensive','slow','difficult','hard to',
    'no way to','broken','frustrat','wast','manual process',
    'time-consuming','complex','unreliable','inaccessible',
    'underserved','unbanked','overpriced','delayed','stuck',
]
_SOLUTION_SIGNALS = [
    'app','platform','tool','service','system','software','marketplace',
    'saas','api','dashboard','website','bot','chatbot','solution','product',
    'connect','automate','enable','simplify','streamline','digitize',
    'ai-powered','using ai','mobile app','web app','mobile application',
    'subscription model','subscription service',
]
_MARKET_GEO = [
    'egypt','cairo','egyptian','giza','alexandria',
    'saudi','riyadh','jeddah','ksa','saudi arabia',
    'uae','dubai','abu dhabi','emirates','united arab',
    'morocco','casablanca','rabat','marrakech',
    'nigeria','lagos','abuja',
    'kenya','nairobi',
    'usa','america','silicon valley','new york','united states',
    'uk','london','britain','england',
    'mena','africa','middle east','gulf','gcc',
    'global','worldwide','international','emerging market',
]
_MARKET_CUSTOMER = [
    'sme','smes','small business','small businesses','enterprise','enterprises',
    'b2b','b2c','d2c','startup','startups',
    'consumer','consumers','user','users','customer','customers','client','clients',
    'patient','patients','student','students',
    'farmer','farmers','freelancer','freelancers',
    'driver','drivers','merchant','merchants','retailer','retailers',
    'hospital','hospitals','clinic','clinics',
    'school','schools','university','universities',
    'individual','family','families','company','companies',
]


def _merge_accumulated_text(previous_text: str, new_text: str) -> str:
    prev = (previous_text or "").strip()
    new = (new_text or "").strip()
    if not prev:
        return new
    if not new or new == prev or new in prev:
        return prev
    return f"{prev} {new}".strip()


def _extract_components(text: str, prev_state: Optional[dict] = None) -> dict:
    t  = text.lower()
    wc = len(t.split())
    prev = prev_state or {}
    has_problem  = prev.get('has_problem', False) or any(s in t for s in _PROBLEM_SIGNALS)
    has_solution = prev.get('has_solution', False) or any(s in t for s in _SOLUTION_SIGNALS)
    has_geo      = any(g in t for g in _MARKET_GEO)
    has_customer = any(c in t for c in _MARKET_CUSTOMER)
    has_market   = prev.get('has_market', False) or has_geo or has_customer
    return {
        'has_problem':    has_problem,
        'has_solution':   has_solution,
        'has_market':     has_market,
        'word_count':     wc,
        'is_substantial': prev.get('is_substantial', False) or wc >= 8,
        'count':          int(prev.get('count', 0)) + 1,
        'accumulated_text': _merge_accumulated_text(prev.get('accumulated_text', ''), text),
    }


def _build_analysis_text(last_user_msg: str, context: dict, messages: list) -> str:
    clarification_state = context.get('clarification_state', {}) or {}
    accumulated = _merge_accumulated_text(
        clarification_state.get('accumulated_text', ''),
        last_user_msg,
    ).strip()
    if accumulated:
        return accumulated

    substantive = []
    for msg in messages:
        if msg.role != 'user':
            continue
        content = msg.content.strip()
        if not content:
            continue
        lower = content.lower()
        if len(lower.split()) <= 4 and lower in _CASUAL_SHORT_SET:
            continue
        if any(lower.startswith(prefix) for prefix in _CASUAL_PREFIXES):
            if not any(sig in lower for sig in _PROBLEM_SIGNALS + _SOLUTION_SIGNALS + _MARKET_GEO + _MARKET_CUSTOMER):
                continue
        substantive.append(content)

    joined = " ".join(substantive).strip()
    return joined or last_user_msg


_GENERIC_IDEA_TOKENS = {
    'fintech', 'healthtech', 'edtech', 'saas', 'ecommerce', 'agritech', 'logistics',
    'money', 'students', 'student', 'college', 'college students', 'app', 'platform',
    'tool', 'software', 'marketplace', 'idea', 'startup', 'product', 'service',
    'b2b', 'b2c', 'consumer', 'consumers', 'users', 'user', 'business', 'businesses',
}
_GENERIC_CUSTOMER_HINTS = {
    'user', 'users', 'customer', 'customers', 'consumer', 'consumers',
    'business', 'businesses', 'company', 'companies', 'individual',
    'family', 'families',
}
_GENERIC_MECHANISM_HINTS = {
    'app', 'platform', 'tool', 'software', 'service', 'product',
    'solution', 'saas', 'marketplace',
}
_EXPLICIT_MECHANISM_SIGNALS = {
    'workflow', 'dashboard', 'api', 'subscription', 'forecasting',
    'planning', 'automation', 'payments', 'underwriting', 'inventory',
    'booking', 'scheduling', 'procurement', 'financing', 'lending',
    'insurance', 'crm', 'erp', 'point of sale',
}

# ═══════════════════════════════════════════════════════════════
# CONVERSATION INTELLIGENCE LAYER
# Classifies every user turn BEFORE any pipeline decision.
# Prevents loops, handles casual conversation, and respects
# explicit user control commands.
# ═══════════════════════════════════════════════════════════════

# Post-decision conversation modes — drive different chat behavior, not just
# different text. Returned by _post_decision_route, consumed by chat builders.
POST_DECISION_MODES = (
    'STANDARD_ADVISOR',     # GO / CONDITIONAL / NO_GO — interpretive advisor mode
    'RESOLVING_CONFLICT',   # CONFLICTING_SIGNALS — must surface conflict + ask resolution
    'ADVISORY_ONLY',        # HIGH_UNCERTAINTY — every reply prefaces with caveat
    'RE_CLARIFY',           # INSUFFICIENT_DATA — re-prompt for missing L1/L3 fields
)


_ROUTING_LOG = __import__('logging').getLogger("midan.routing")

# Decision states the chat layer is allowed to act on as a "post-decision"
# routing input. Anything outside this set means the L4 envelope is missing
# or pre-analysis — chat must NOT render an opener that exposes UNKNOWN.
_VALID_DECISION_STATES = {
    'GO', 'CONDITIONAL', 'NO_GO',
    'INSUFFICIENT_DATA', 'HIGH_UNCERTAINTY', 'CONFLICTING_SIGNALS',
}


def _post_decision_route(context: dict) -> dict:
    """
    Map a prior decision_state into a conversation mode + the L4 evidence the
    chat layer should reference. Single source of behavioral truth for what
    happens after a decision was rendered.

    Strict guard: if decision_state is missing, empty, or not one of the
    documented states (GO/CONDITIONAL/NO_GO/INSUFFICIENT_DATA/HIGH_UNCERTAINTY/
    CONFLICTING_SIGNALS) — force RE_CLARIFY so we never expose `UNKNOWN` or
    `unknown strength` to the user. A missing decision_state at the chat layer
    is a routing-correctness issue and is logged as such.
    """
    raw_state = context.get('decision_state')
    state = (raw_state or '').upper()

    if state not in _VALID_DECISION_STATES:
        _ROUTING_LOG.warning(
            "[ROUTING] _post_decision_route: invalid/missing decision_state "
            "(raw=%r) — forcing RE_CLARIFY to avoid UNKNOWN leak",
            raw_state,
        )
        return {
            'mode': 'RE_CLARIFY',
            'state': raw_state or None,
            'reason': 'missing_or_invalid_decision_state_in_context',
        }

    # The L4 envelope is the canonical post-analysis payload. Pull it out once.
    l4 = context.get('l4_decision') or {}

    if state == 'CONFLICTING_SIGNALS':
        unresolved = [
            c for c in l4.get('conflicting_signals', [])
            if c.get('severity') == 'high' and c.get('resolution_required')
        ]
        return {
            'mode': 'RESOLVING_CONFLICT',
            'state': state,
            'unresolved_conflicts': unresolved,
            'reason': 'last decision halted on high-severity unresolved conflict(s)',
        }

    if state == 'HIGH_UNCERTAINTY':
        quality = l4.get('decision_quality', {})
        return {
            'mode': 'ADVISORY_ONLY',
            'state': state,
            'uncertainty_basis': {
                'input_completeness': (quality.get('input_completeness') or {}).get('basis'),
                'signal_agreement':   (quality.get('signal_agreement')   or {}).get('basis'),
                'overall_uncertainty': quality.get('overall_uncertainty'),
            },
            'reason': 'last decision flagged as advisory due to high overall uncertainty',
        }

    if state == 'INSUFFICIENT_DATA':
        return {
            'mode': 'RE_CLARIFY',
            'state': state,
            'reason': 'last attempt halted because required L1/L3 fields were unavailable',
        }

    # GO / CONDITIONAL / NO_GO / unknown → standard advisor mode
    return {
        'mode': 'STANDARD_ADVISOR',
        'state': state or None,
        'reason': 'normal post-decision interpretive mode',
    }


def _classify_intent(text: str, context: dict, messages: list) -> dict:
    """
    Classify user message into one of 5 intent types.

    Priority (evaluated top-down, first match wins):
      OVERRIDE_COMMAND → user explicitly demands analysis ("analyze now", "run it")
      CLARIFICATION    → post-analysis turn, route to advisor
      CASUAL           → personal intro / ack / small talk, no idea content
      ANALYSIS_REQUEST → enough signals to run analysis immediately
      PARTIAL_IDEA     → has some idea signals but missing critical components

    Anti-loop: if ≥ 2 prior non-vague user turns, forces ANALYSIS_REQUEST
    to prevent repetitive questioning.

    Returns: {intent, should_analyze, reason}
    """
    t  = text.lower().strip()
    wc = len(t.split())

    # ── 2. Post-analysis clarification (checked BEFORE override) ─────────────
    # Detection is via L4 decision_state — the canonical post-decision marker.
    # Falls back to legacy tas_score for back-compat with frontends that haven't
    # migrated to the L4 envelope yet.
    has_l4_decision  = bool(context.get('decision_state'))
    has_legacy_tas   = bool(context.get('tas_score'))
    if has_l4_decision or has_legacy_tas:
        return {
            'intent': 'CLARIFICATION',
            'should_analyze': False,
            'reason': 'post_analysis',
            'post_decision_state': context.get('decision_state'),
        }

    # ── 3. Casual / personal — no idea content ───────────────────────────────
    is_personal_prefix = any(t.startswith(p) for p in _CASUAL_PREFIXES)
    is_short_ack       = wc <= 4 and t in _CASUAL_SHORT_SET
    no_idea_signals    = (
        not any(s in t for s in _PROBLEM_SIGNALS)
        and not any(s in t for s in _SOLUTION_SIGNALS)
        and not any(g in t for g in _MARKET_GEO)
        and not any(s in t for s in SECTOR_KEYWORDS.get('fintech', []) + SECTOR_KEYWORDS.get('saas', []))
    )
    if (is_personal_prefix or is_short_ack) and no_idea_signals:
        return {'intent': 'CASUAL', 'should_analyze': False, 'reason': 'personal_conversation'}

    # ── 4. Extract idea components (problem, solution, market) ────────────────
    comps = _extract_components(text, context.get('clarification_state'))
    # Strict completeness gate — minimum 2 of 3 components must be present in
    # the accumulated conversation. Below this, NO override and NO anti-loop
    # trigger can force analysis. The L1 confidence gate inside process_idea
    # is the second line of defence; this is the first.
    components_present = sum([
        comps['has_problem'], comps['has_solution'], comps['has_market'],
    ])
    accumulated_wc = len(comps.get('accumulated_text', '').split())
    has_minimum_completeness = components_present >= 2 and accumulated_wc >= 10

    # ── 1. Override command — only honoured AFTER completeness is met ────────
    is_override = (
        any(t == cmd or t.startswith(cmd + ' ') or t.endswith(' ' + cmd) for cmd in _OVERRIDE_COMMANDS)
        or (wc <= 5 and any(cmd in t for cmd in ['analyze', 'run it', 'go ahead', 'break it down', 'just go']))
    )
    if is_override:
        if has_minimum_completeness:
            return {'intent': 'OVERRIDE_COMMAND', 'should_analyze': True, 'reason': 'explicit_trigger'}
        return {'intent': 'PARTIAL_IDEA', 'should_analyze': False,
                'reason': 'override_blocked_insufficient_signals'}

    # ── Anti-loop guard: after ≥2 substantive partial turns, force analysis
    # ONLY if completeness is now met. Never analyse with < 2 components.
    user_msgs = [m for m in messages if m.role == 'user']
    prior_partial = sum(
        1 for m in user_msgs[:-1]
        if len(m.content.split()) >= 5
        and not any(m.content.lower().strip().startswith(v) for v in _VAGUE_STARTERS)
    )
    if prior_partial >= 2 and has_minimum_completeness:
        return {'intent': 'ANALYSIS_REQUEST', 'should_analyze': True, 'reason': 'anti_loop_forced'}

    # ── 5. Sufficient signals for immediate analysis ──────────────────────────
    # Stricter than before — 2-of-3 components AND ≥10 words. Pure word-count
    # fallback removed: a 15-word vague rant must still go through clarification.
    if has_minimum_completeness:
        return {'intent': 'ANALYSIS_REQUEST', 'should_analyze': True, 'reason': 'sufficient_signals'}

    # ── 6. Partial — some signals, single follow-up needed ───────────────────
    return {'intent': 'PARTIAL_IDEA', 'should_analyze': False, 'reason': 'insufficient_signals'}


def _casual_response(text: str) -> str:
    """Natural reply for casual / personal messages. Never triggers the pipeline."""
    t = text.lower().strip()

    # Name introduction
    for prefix in ('call me ', 'my name is '):
        if t.startswith(prefix):
            name = t[len(prefix):].strip().split()[0].capitalize()
            return f"Got it, {name}. What are you working on?"

    # Location / background intro
    for prefix in ("i'm from ", 'i am from ', 'from '):
        if t.startswith(prefix):
            place = text[len(prefix):].strip().split()[0].capitalize()
            return f"{place} — that's a live market for a few sectors. What's the idea?"

    for prefix in ("i'm a ", 'i am a ', 'i studied '):
        if t.startswith(prefix):
            return "Good context. What startup idea are you pressure-testing?"

    # Acknowledgments
    if t in {'ok', 'okay', 'cool', 'nice', 'got it', 'understood', 'noted', 'makes sense', 'i see', 'right', 'alright'}:
        return "What's the idea you're working on?"
    if t in {'thanks', 'thank you', 'ty', 'thx', 'thanks again'}:
        return "Anytime. What else do you want to pressure-test?"
    if t in {'yes', 'sure', 'yep', 'agreed', 'perfect', 'great', 'awesome'}:
        return "Tell me the idea and I'll run the numbers."
    if t in {'no', 'nope', 'not yet', 'maybe', 'later'}:
        return "Whenever you're ready — drop the idea and I'll analyze it."

    # Generic casual
    return "What are you building? Drop the idea and I'll run a full analysis."


def _smart_followup(text: str, messages: list, context_state: Optional[dict] = None) -> str:
    """
    Ask ONE targeted follow-up question for a PARTIAL_IDEA turn.
    Tracks prior questions to avoid repetition.
    Never asks more than one question per turn.
    """
    comps = _extract_components(text, context_state)

    # Track what has already been asked
    asked: set = set()
    for m in messages:
        if m.role != 'assistant':
            continue
        mc = m.content.lower()
        if any(w in mc for w in ('problem', 'pain', 'what exactly', 'what specific')):
            asked.add('problem')
        if any(w in mc for w in ('market', 'geography', 'country', 'who exactly', 'customer type', 'which region')):
            asked.add('market')
        if any(w in mc for w in ('approach', 'mechanism', 'app', 'marketplace', 'saas', 'platform', 'how does it work')):
            asked.add('solution')

    # Ask about the most important missing component not yet asked about.
    # Tone: a consultant guiding the conversation, not a form filler.
    if not comps['has_problem'] and 'problem' not in asked:
        return (
            "Walk me through the actual pain — what specifically goes wrong today, "
            "and for whom?"
        )
    if not comps['has_market'] and 'market' not in asked:
        return (
            "Where does this live — country and customer type? "
            "Could be Egyptian SMEs, UAE consumers, restaurants in Riyadh, whatever fits."
        )
    if not comps['has_solution'] and 'solution' not in asked:
        return (
            "How does it actually work — a marketplace, a SaaS tool, a service, an app? "
            "One line is enough."
        )

    # All targeted questions already asked — say plainly what's still missing
    # so the conversation moves forward instead of looping.
    still_missing = []
    if not comps['has_problem']:
        still_missing.append("the specific problem")
    if not comps['has_market']:
        still_missing.append("the market (country and customer type)")
    if not comps['has_solution']:
        still_missing.append("the mechanism (SaaS / marketplace / service / app)")
    if still_missing:
        return (
            "I still need " + ", ".join(still_missing) +
            " before I can run a real read. Two sentences covering them is enough."
        )
    return (
        "Round it out a bit — sector, market, and how it works. Two sentences is enough "
        "and we can move."
    )


_OP_REPLY_LOG = __import__('logging').getLogger("midan.operator_reply")





# Export everything defined in this module — including underscore-prefixed
# helpers — so other midan submodules can wildcard-import the full surface.
__all__ = [name for name in list(globals().keys()) if not name.startswith('__')]
