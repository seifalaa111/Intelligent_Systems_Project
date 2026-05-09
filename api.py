"""
api.py — re-export shim.

The MIDAN implementation has been split into the `midan/` package. This
file preserves the legacy `import api` interface so tests, frontend, and
any external consumer keep working without modification. Public symbols
are re-exported here verbatim from the relevant midan submodule.

For new code, prefer importing directly from the submodule that owns the
symbol — e.g. `from midan.l4_decision import compute_l4_decision`.

Layer modules (see midan/__init__.py for the full description):
    midan.core              — schemas, constants, ML loaders, utilities
    midan.l0_gate           — input validation gate
    midan.l1_parser         — confidence-scored feature extraction
    midan.l2_intelligence   — macro / regime / FCM / SHAP / freshness
    midan.l3_reasoning      — structured idea reasoning
    midan.l4_decision       — decision state machine
    midan.conversation      — intent classification + post-decision routing
    midan.response          — payload builder + chat + operator + projection
    midan.pipeline          — process_idea + run_inference orchestrators
    midan.endpoints         — FastAPI app + HTTP endpoints
"""

# ── core: schemas, constants, ML loaders, utilities ─────────────────────────
from midan.core import *  # noqa: F401,F403

# ── L0 gate ─────────────────────────────────────────────────────────────────
from midan.l0_gate import *  # noqa: F401,F403

# ── L1 parser ───────────────────────────────────────────────────────────────
from midan.l1_parser import *  # noqa: F401,F403

# ── L2 intelligence ─────────────────────────────────────────────────────────
from midan.l2_intelligence import *  # noqa: F401,F403

# ── L3 reasoning ────────────────────────────────────────────────────────────
from midan.l3_reasoning import *  # noqa: F401,F403

# ── L4 decision engine ──────────────────────────────────────────────────────
from midan.l4_decision import *  # noqa: F401,F403

# ── Conversation layer ──────────────────────────────────────────────────────
from midan.conversation import *  # noqa: F401,F403

# ── Response layer (chat, operator, projection, builder) ────────────────────
from midan.response import *  # noqa: F401,F403

# ── Pipeline orchestrators ──────────────────────────────────────────────────
from midan.pipeline import *  # noqa: F401,F403

# ── Endpoints + FastAPI app instance ────────────────────────────────────────
# `api` (the FastAPI app) is exposed by midan.endpoints; importing * here
# pulls in the app instance plus all six route handlers. Tests that do
# `TestClient(api.api)` continue to work.
from midan.endpoints import *  # noqa: F401,F403
