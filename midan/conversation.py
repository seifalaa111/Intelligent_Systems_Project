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

# ── we use these prefixes to detect casual / personal turns with no idea content ──
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

# ── we treat these as override commands when the user explicitly demands immediate analysis ──
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

# we define post-decision conversation modes here — they drive different chat behavior,
# not just different text; returned by _post_decision_route and consumed by chat builders
POST_DECISION_MODES = (
    'STANDARD_ADVISOR',     # GO / CONDITIONAL / NO_GO — interpretive advisor mode
    'RESOLVING_CONFLICT',   # CONFLICTING_SIGNALS — must surface conflict + ask resolution
    'ADVISORY_ONLY',        # HIGH_UNCERTAINTY — every reply prefaces with caveat
    'RE_CLARIFY',           # INSUFFICIENT_DATA — re-prompt for missing L1/L3 fields
)


_ROUTING_LOG = __import__('logging').getLogger("midan.routing")
_TRACE_LOG   = __import__('logging').getLogger("midan.trace")
if not _TRACE_LOG.handlers:
    import logging as _logging_c, pathlib as _pathlib_c
    _trace_path_c = _pathlib_c.Path(__file__).parent.parent / "trace.log"
    _trace_fh_c = _logging_c.FileHandler(str(_trace_path_c), mode="a", encoding="utf-8")
    _trace_fh_c.setFormatter(_logging_c.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _TRACE_LOG.addHandler(_trace_fh_c)
    _TRACE_LOG.setLevel(_logging_c.DEBUG)
    _TRACE_LOG.propagate = False

# we enumerate valid post-decision states the chat layer is allowed to act on;
# anything outside this set means the L4 envelope is missing or pre-analysis —
# we must NOT render an opener that exposes UNKNOWN to the user
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

    # we pull the L4 envelope out once — it's the canonical post-analysis payload
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

    # GO / CONDITIONAL / NO_GO / unknown → we fall through to standard advisor mode
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
    # we detect post-analysis turns via L4 decision_state — that's the canonical post-decision marker;
    # we fall back to legacy tas_score for frontends that haven't migrated to the L4 envelope yet
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
        _TRACE_LOG.info("[TRACE][_classify_intent] → CASUAL (is_personal_prefix=%s, is_short_ack=%s)", is_personal_prefix, is_short_ack)
        return {'intent': 'CASUAL', 'should_analyze': False, 'reason': 'personal_conversation'}

    # ── 4. Extract idea components (problem, solution, market) ────────────────
    comps = _extract_components(text, context.get('clarification_state'))
    # we enforce a strict completeness gate here — minimum 2 of 3 components must be present
    # in the accumulated conversation; below this, no override and no anti-loop trigger can force
    # analysis; the L1 confidence gate inside process_idea is the second line of defence, this is the first
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

    # ── we added this anti-loop guard to force analysis after 2+ substantive partial turns,
    # but ONLY if completeness is met — we never analyse with < 2 components
    user_msgs = [m for m in messages if m.role == 'user']
    prior_partial = sum(
        1 for m in user_msgs[:-1]
        if len(m.content.split()) >= 5
        and not any(m.content.lower().strip().startswith(v) for v in _VAGUE_STARTERS)
    )
    if prior_partial >= 2 and has_minimum_completeness:
        return {'intent': 'ANALYSIS_REQUEST', 'should_analyze': True, 'reason': 'anti_loop_forced'}

    # ── 5. Sufficient signals for immediate analysis ──────────────────────────
    # we made this stricter — 2-of-3 components AND ≥10 words are both required;
    # we removed the pure word-count fallback: a 15-word vague rant must still go through clarification
    if has_minimum_completeness:
        return {'intent': 'ANALYSIS_REQUEST', 'should_analyze': True, 'reason': 'sufficient_signals'}

    # ── 6. Partial — some signals, single follow-up needed ───────────────────
    _TRACE_LOG.info("[TRACE][_classify_intent] → PARTIAL_IDEA (comps_present=%d, accum_wc=%d)",
                    sum([comps['has_problem'], comps['has_solution'], comps['has_market']]),
                    len(comps.get('accumulated_text', '').split()))
    return {'intent': 'PARTIAL_IDEA', 'should_analyze': False, 'reason': 'insufficient_signals'}


def _casual_response(text: str) -> str:
    """
    Heuristic fallback for casual / personal messages when Gemini is unavailable.
    Persona: direct, dry, peer-to-peer. No corporate-speak.
    """
    t = text.lower().strip()
    _TRACE_LOG.warning("[TRACE][_casual_response] LLM SKIPPED — heuristic casual | input=%r", text[:80])

    for prefix in ('call me ', 'my name is '):
        if t.startswith(prefix):
            name = t[len(prefix):].strip().split()[0].capitalize()
            result = f"Got it, {name}. What are you working on?"
            _TRACE_LOG.warning("[TRACE][_casual_response] FINAL STRING: %r", result)
            return result

    for prefix in ("i'm from ", 'i am from ', 'from '):
        if t.startswith(prefix):
            place = text[len(prefix):].strip().split()[0].capitalize()
            result = f"{place} — active market. What's the idea?"
            _TRACE_LOG.warning("[TRACE][_casual_response] FINAL STRING: %r", result)
            return result

    for prefix in ("i'm a ", 'i am a ', 'i studied '):
        if t.startswith(prefix):
            _TRACE_LOG.warning("[TRACE][_casual_response] FINAL STRING: 'Good context. What are you building?'")
            return "Good context. What are you building?"

    if t in {'ok', 'okay', 'cool', 'nice', 'got it', 'understood', 'noted', 'makes sense', 'i see', 'right', 'alright'}:
        _TRACE_LOG.warning("[TRACE][_casual_response] FINAL STRING: 'What's the idea?'")
        return "What's the idea?"
    if t in {'thanks', 'thank you', 'ty', 'thx', 'thanks again'}:
        _TRACE_LOG.warning("[TRACE][_casual_response] FINAL STRING: 'Anytime...'")
        return "Anytime. Anything else to pressure-test?"
    if t in {'yes', 'sure', 'yep', 'agreed', 'perfect', 'great', 'awesome'}:
        _TRACE_LOG.warning("[TRACE][_casual_response] FINAL STRING: 'Drop the idea...'")
        return "Drop the idea and I'll run it."
    if t in {'no', 'nope', 'not yet', 'maybe', 'later'}:
        _TRACE_LOG.warning("[TRACE][_casual_response] FINAL STRING: 'Whenever you're ready.'")
        return "Whenever you're ready."

    _TRACE_LOG.warning("[TRACE][_casual_response] FINAL STRING: 'What are you building?' (default)")
    return "What are you building?"


def _smart_followup(text: str, messages: list, context_state: Optional[dict] = None) -> str:
    """
    Context-aware heuristic follow-up for the pre-analysis phase.
    Tracks what the assistant already asked so it never repeats the same question.
    Returns exactly ONE targeted question — the next missing piece.

    The tracking patterns must match the exact strings this function returns,
    plus any strings the Gemini system prompt might generate. Keep them in sync.
    """
    _TRACE_LOG.warning("[TRACE][_smart_followup] LLM SKIPPED — heuristic followup | input=%r", text[:80])
    comps = _extract_components(text, context_state)

    # Scan prior assistant messages to detect what was already asked.
    # Patterns must be broad enough to match both this function's own output
    # AND Gemini's paraphrases of the same question.
    asked: set = set()
    for m in messages:
        if m.role != 'assistant':
            continue
        mc = m.content.lower()
        # problem-related patterns
        if any(w in mc for w in (
            'problem', 'pain', 'breaks', 'broken', 'goes wrong', 'what exactly',
            'for whom', 'who has', 'who specifically has', 'the thing that',
        )):
            asked.add('problem')
        # market-related patterns
        if any(w in mc for w in (
            'country', 'geography', 'market', 'customer type', 'which region',
            'type of customer', 'which country', 'who specifically', 'geography',
        )):
            asked.add('market')
        # solution-related patterns
        if any(w in mc for w in (
            'how does it work', 'how does it actually', 'mechanism',
            'marketplace', 'saas', 'service', 'platform',
            'how does this work', 'how are you solving',
        )):
            asked.add('solution')

    if not comps['has_problem'] and 'problem' not in asked:
        _TRACE_LOG.warning("[TRACE][_smart_followup] FINAL STRING: 'What specifically breaks today — and for whom?'")
        return "What specifically breaks today — and for whom?"
    if not comps['has_market'] and 'market' not in asked:
        _TRACE_LOG.warning("[TRACE][_smart_followup] FINAL STRING: 'Which country, and what type of customer?'")
        return "Which country, and what type of customer?"
    if not comps['has_solution'] and 'solution' not in asked:
        _TRACE_LOG.warning("[TRACE][_smart_followup] FINAL STRING: 'How does it work — marketplace, SaaS, service, app?'")
        return "How does it work — marketplace, SaaS, service, app?"

    # All three queued — list what's still missing so conversation moves forward
    still_missing = []
    if not comps['has_problem']:
        still_missing.append("the problem")
    if not comps['has_market']:
        still_missing.append("the market")
    if not comps['has_solution']:
        still_missing.append("the approach")
    if still_missing:
        result = "Still need " + " and ".join(still_missing) + " — two sentences is enough."
        _TRACE_LOG.warning("[TRACE][_smart_followup] FINAL STRING: %r", result)
        return result
    _TRACE_LOG.warning("[TRACE][_smart_followup] FINAL STRING: 'Two sentences on the market and approach and we can move.'")
    return "Two sentences on the market and approach and we can move."


_OP_REPLY_LOG = __import__('logging').getLogger("midan.operator_reply")





# we export everything defined in this module — including underscore-prefixed helpers —
# so other midan submodules can wildcard-import the full surface
__all__ = [name for name in list(globals().keys()) if not name.startswith('__')]
