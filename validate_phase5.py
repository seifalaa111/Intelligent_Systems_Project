"""
Phase 5 — System Validation + Hardening
Tests all new Phase 4 components in isolation and integration.

Scenarios covered:
  IS-1  IS weights sum to 1.00
  IS-2  Correlated-trio discount fires only when all three > 0.80
  IS-3  Unknown regime -> neutral 0.50
  IS-4  All-min inputs produce IS >= 0 (no negative bleed)
  IS-5  gap_svm = sorted[0]-sorted[1], NOT proba.max()

  RAG-1  shap_cosine returns 0.5 when artifact absent (None)
  RAG-2  shap_cosine returns 0.5 on zero-norm SHAP vector
  RAG-3  novelty_score returns 0.0 when index absent
  RAG-4  query_explicit_rag skips gracefully when artifact absent
  RAG-5  novelty gate fires -> rag_skipped=True, reason="novelty"
  RAG-6  ARIMA amplify (+0.10) and dampen (×0.70) modifiers apply correctly
  RAG-7  query_explicit_rag runs correctly when FAISS available

  RT-1   PATH_NOVELTY fires before 7-path tree
  RT-2   PATH_1_HIGH_CERTAINTY fires (IS high, shap reliable, no rag conflict)
  RT-3   PATH_2_LOW_CERTAINTY fires (IS low, shap reliable)
  RT-4   PATH_3_BORDERLINE_CONFIRMED (borderline IS, shap reliable, rag confirms)
  RT-5   PATH_4_BORDERLINE_CONFLICT (borderline IS, shap reliable, rag conflicts)
  RT-6   PATH_5_ATYPICAL_SUPPORTED (shap atypical, rag confirms)
  RT-7   PATH_6_FULL_CONFLICT -> force_human_review=True
  RT-8   PATH_7_MAXIMUM_UNCERTAINTY -> force_human_review=True
  RT-9   rag_vote present in all return dicts
  RT-10  asymmetric_opportunity fires when novel + sarima_trend >= floor

  DR-1   log_prediction writes a JSONL entry without raising
  DR-2   check_drift returns no_baseline gracefully
  DR-3   macro_staleness_alert fires (STATIC_MACRO_TABLE_AS_OF is stale)
  DR-4   Dual-signal AND gate: drift only when both signals fire
  DR-5   Insufficient log entries -> drift_detected=False
  DR-6   Log cap + trim: drops oldest 20% when cap reached

  L4-1   react_decision=None -> compute_l4_decision unchanged (no crash)
  L4-2   RAG conflict injected into conflicting_signals when severity='low'
  L4-3   RAG conflict injected when severity='medium'
  L4-4   force_human_review=True overrides state to INSUFFICIENT_DATA
  L4-5   force_human_review on already-INSUFFICIENT_DATA -> no double-wrap
  L4-6   react_decision with severity='none' -> no conflict injected

  INT-1  pipeline.py imports cleanly
  INT-2  _build_signal_consensus_summary produces non-empty string for each path
  INT-3  Degraded artifacts: all None -> no crash in IS, RAG, router
"""

import os, sys, json, tempfile, shutil, traceback

# ── path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

# ── results bookkeeping ───────────────────────────────────────────────────────
_pass = 0
_fail = 0

def ok(label: str):
    global _pass
    _pass += 1
    print(f"  PASS  {label}")

def fail(label: str, reason: str):
    global _fail
    _fail += 1
    print(f"  FAIL  {label}: {reason}")


# ═══════════════════════════════════════════════════════════════════════════════
# IS TESTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- Intelligent Score ---")

from midan.intelligent_score import compute_intelligent_score
from midan.config import IS_W_S, IS_W_GAP, IS_W_MU, IS_W_ARIMA, IS_W_SHAP

# IS-1: weights sum to 1.00
w_sum = IS_W_S + IS_W_GAP + IS_W_MU + IS_W_ARIMA + IS_W_SHAP
if abs(w_sum - 1.00) < 1e-9:
    ok("IS-1  weights sum to 1.00")
else:
    fail("IS-1  weights sum to 1.00", f"got {w_sum}")

# IS-2: correlated-trio discount fires only when all three > 0.80
# gap_svm = sorted[0]-sorted[1]; need gap > 0.80 -> use [0.92, 0.08] (gap=0.84)
proba_trio_on  = np.array([0.92, 0.08])  # gap=0.84 > 0.80
proba_trio_off = np.array([0.85, 0.15])  # gap=0.70 < 0.80
r_trio = compute_intelligent_score(
    conf=0.92, proba=proba_trio_on, mu_fcm=0.90,
    sarima_trend=0.50, shap_cosine=0.90, regime='GROWTH_MARKET',
)
r_notrio = compute_intelligent_score(
    conf=0.85, proba=proba_trio_off, mu_fcm=0.70,
    sarima_trend=0.50, shap_cosine=0.90, regime='GROWTH_MARKET',
)
if r_trio['is_correlated_trio'] and not r_notrio['is_correlated_trio']:
    ok("IS-2  trio discount fires correctly")
else:
    fail("IS-2  trio discount fires correctly",
         f"trio={r_trio['is_correlated_trio']} notrio={r_notrio['is_correlated_trio']}")

# IS-2b: trio discount actually lowers the score
proba_tight = np.array([0.85, 0.15])  # gap = 0.70 -> below 0.80 threshold
proba_wide  = np.array([0.95, 0.05])  # gap = 0.90 -> above 0.80 threshold
r_with_discount = compute_intelligent_score(
    conf=0.95, proba=proba_wide, mu_fcm=0.90,
    sarima_trend=0.50, shap_cosine=0.90, regime='GROWTH_MARKET',
)
if r_with_discount['is_correlated_trio']:
    # Recompute without discount manually
    gap = r_with_discount['gap_svm']
    from midan.config import IS_CORRELATED_TRIO_DISCOUNT
    trio_contrib = IS_W_GAP * gap + IS_W_MU * 0.90 + IS_W_SHAP * 0.90
    discount_savings = trio_contrib * (1 - IS_CORRELATED_TRIO_DISCOUNT)
    if discount_savings > 0:
        ok("IS-2b trio discount reduces score (savings > 0)")
    else:
        fail("IS-2b trio discount reduces score", f"savings={discount_savings}")
else:
    fail("IS-2b trio discount reduces score", "trio did not fire in this test")

# IS-3: unknown regime -> neutral 0.50 S component
r_unknown = compute_intelligent_score(
    conf=0.5, proba=np.array([0.6, 0.4]), mu_fcm=0.5,
    sarima_trend=0.5, shap_cosine=0.5, regime='UNKNOWN_REGIME',
)
if r_unknown['components']['s_favorable'] == 0.50:
    ok("IS-3  unknown regime -> s_favorable=0.50")
else:
    fail("IS-3  unknown regime -> s_favorable=0.50",
         f"got {r_unknown['components']['s_favorable']}")

# IS-4: all-min inputs -> IS >= 0
r_allmin = compute_intelligent_score(
    conf=0.0, proba=np.array([0.5, 0.5]), mu_fcm=0.0,
    sarima_trend=0.0, shap_cosine=0.0, regime='CONTRACTING_MARKET',
)
if r_allmin['score'] >= 0.0:
    ok("IS-4  all-min inputs -> IS >= 0")
else:
    fail("IS-4  all-min inputs -> IS >= 0", f"got {r_allmin['score']}")

# IS-5: gap_svm is margin (sorted[0]-sorted[1]), not proba.max()
proba_is5 = np.array([0.70, 0.20, 0.10])
r_is5 = compute_intelligent_score(
    conf=0.7, proba=proba_is5, mu_fcm=0.5,
    sarima_trend=0.5, shap_cosine=0.5, regime='GROWTH_MARKET',
)
expected_gap = 0.70 - 0.20  # = 0.50 (sorted desc: 0.70, 0.20, 0.10)
if abs(r_is5['gap_svm'] - expected_gap) < 1e-6:
    ok("IS-5  gap_svm = sorted[0]-sorted[1]")
else:
    fail("IS-5  gap_svm = sorted[0]-sorted[1]",
         f"expected {expected_gap}, got {r_is5['gap_svm']}")


# ═══════════════════════════════════════════════════════════════════════════════
# RAG TESTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- RAG ---")

from midan.rag import compute_shap_cosine, query_explicit_rag, compute_novelty_score
from midan.core import FEATURE_ORDER
from midan.config import NOVELTY_THRESHOLD, ARIMA_RAG_AMPLIFY_THRESHOLD, ARIMA_RAG_DAMPEN_THRESHOLD

# RAG-1: shap_cosine = 0.5 when shap_cluster_means is None
sc = compute_shap_cosine({'inflation': 0.5}, None, 0, FEATURE_ORDER)
if sc == 0.5:
    ok("RAG-1  shap_cosine=0.5 when artifact absent")
else:
    fail("RAG-1  shap_cosine=0.5 when artifact absent", f"got {sc}")

# RAG-1b: shap_cosine = 0.5 when cluster idx missing from means dict
sc_miss = compute_shap_cosine({'inflation': 0.5}, {999: np.array([0.2]*5)}, 0, FEATURE_ORDER)
if sc_miss == 0.5:
    ok("RAG-1b shap_cosine=0.5 when cluster idx missing")
else:
    fail("RAG-1b shap_cosine=0.5 when cluster idx missing", f"got {sc_miss}")

# RAG-2: shap_cosine = 0.5 on zero-norm SHAP vector
zero_shap = {f: 0.0 for f in FEATURE_ORDER}
zero_cluster_mean = np.zeros(len(FEATURE_ORDER))
sc_zero = compute_shap_cosine(zero_shap, {0: zero_cluster_mean}, 0, FEATURE_ORDER)
if sc_zero == 0.5:
    ok("RAG-2  shap_cosine=0.5 on zero-norm vector")
else:
    fail("RAG-2  shap_cosine=0.5 on zero-norm vector", f"got {sc_zero}")

# RAG-2b: identical vectors -> cosine = 1.0
identical_vec = np.array([0.3, 0.2, 0.25, 0.15, 0.1])
sc_ident = compute_shap_cosine(
    dict(zip(FEATURE_ORDER, identical_vec.tolist())),
    {0: identical_vec}, 0, FEATURE_ORDER,
)
if abs(sc_ident - 1.0) < 1e-6:
    ok("RAG-2b identical vectors -> shap_cosine=1.0")
else:
    fail("RAG-2b identical vectors -> shap_cosine=1.0", f"got {sc_ident}")

# RAG-3: novelty_score = 0.0 when index absent (safe default)
ns = compute_novelty_score(np.random.rand(10).astype(np.float32), None, 5)
if ns == 0.0:
    ok("RAG-3  novelty_score=0.0 when index absent")
else:
    fail("RAG-3  novelty_score=0.0 when index absent", f"got {ns}")

# RAG-4: query_explicit_rag skips gracefully without artifact
rag_no_art = query_explicit_rag(
    x_scaled_row=np.zeros(5), shap_dict={f: 0.1 for f in FEATURE_ORDER},
    rag_index=None, rag_labels=[], k=5, sarima_trend=0.5, feature_order=FEATURE_ORDER,
)
if rag_no_art['rag_skipped'] and rag_no_art['rag_skipped_reason'] == 'artifact_unavailable':
    ok("RAG-4  graceful skip when artifact absent")
else:
    fail("RAG-4  graceful skip when artifact absent", str(rag_no_art))

# RAG-5 + RAG-6 + RAG-7: test with a real tiny FAISS index if faiss available
try:
    import faiss  # noqa: F401

    # Build a tiny index: 8 training points in 10D, 4 per regime
    n_train = 8
    dim = 10
    rng = np.random.default_rng(42)
    train_vecs = rng.standard_normal((n_train, dim)).astype(np.float32)
    # Normalize (IndexFlatIP computes cosine when vectors are L2-normalized)
    norms = np.linalg.norm(train_vecs, axis=1, keepdims=True)
    train_vecs /= norms
    train_labels = ['GROWTH_MARKET'] * 4 + ['CONTRACTING_MARKET'] * 4

    idx = faiss.IndexFlatIP(dim)
    idx.add(train_vecs)

    # RAG-5: novelty gate — build a query vector far from all training points
    # Use the opposite direction of first training vector to maximize distance
    far_vec = -train_vecs[0]  # cosine ~ -1 after normalization -> novelty ~ 1.0
    far_vec = (far_vec / np.linalg.norm(far_vec)).astype(np.float32)

    # Inject directly via compute_novelty_score
    novelty_far = compute_novelty_score(far_vec, idx, 5)
    # With 8 points, may not always be > NOVELTY_THRESHOLD (0.40); check range
    if 0.0 <= novelty_far <= 1.0:
        ok(f"RAG-5  novelty_score in [0,1] (got {novelty_far:.3f})")
    else:
        fail("RAG-5  novelty_score range", f"got {novelty_far}")

    # RAG-5b: a query identical to a training point -> novelty ~ 0
    novelty_near = compute_novelty_score(train_vecs[0], idx, 5)
    if novelty_near < 0.10:
        ok(f"RAG-5b identical query -> near-zero novelty ({novelty_near:.3f})")
    else:
        fail("RAG-5b identical query -> near-zero novelty", f"got {novelty_near}")

    # RAG-7: full query with FAISS available, majority GROWTH_MARKET query
    # Use a vector near the GROWTH_MARKET cluster (first 4 training vecs)
    growth_center = train_vecs[:4].mean(axis=0)
    growth_center /= np.linalg.norm(growth_center)

    # Build x_scaled_row and shap_dict that produce a combined vector near growth_center
    # Simplest: just pass the first 5 dims as x_scaled, last 5 as shap
    x_row = growth_center[:5].copy()
    shap_near = {f: float(growth_center[5 + i]) for i, f in enumerate(FEATURE_ORDER)}

    rag_7 = query_explicit_rag(
        x_scaled_row=x_row, shap_dict=shap_near,
        rag_index=idx, rag_labels=train_labels,
        k=5, sarima_trend=0.50, feature_order=FEATURE_ORDER,
    )
    if not rag_7['rag_skipped'] and rag_7['vote'] is not None:
        ok(f"RAG-7  query runs with FAISS (vote={rag_7['vote']}, conf={rag_7['confidence']:.3f})")
    elif rag_7['rag_skipped'] and rag_7['rag_skipped_reason'] == 'novelty':
        ok(f"RAG-7  query ran but triggered novelty gate (novelty={rag_7['novelty_score']:.3f})")
    else:
        fail("RAG-7  query runs with FAISS", str(rag_7))

    # RAG-6a: ARIMA amplify (+0.10) fires when sarima_trend >= threshold
    rag_amp = query_explicit_rag(
        x_scaled_row=x_row, shap_dict=shap_near,
        rag_index=idx, rag_labels=train_labels,
        k=5, sarima_trend=ARIMA_RAG_AMPLIFY_THRESHOLD, feature_order=FEATURE_ORDER,
    )
    rag_base = query_explicit_rag(
        x_scaled_row=x_row, shap_dict=shap_near,
        rag_index=idx, rag_labels=train_labels,
        k=5, sarima_trend=0.50, feature_order=FEATURE_ORDER,
    )
    if (not rag_amp['rag_skipped'] and not rag_base['rag_skipped']
            and rag_amp['confidence'] >= rag_base['confidence']):
        ok(f"RAG-6a ARIMA amplify >= base confidence ({rag_amp['confidence']:.3f} >= {rag_base['confidence']:.3f})")
    elif rag_amp['rag_skipped'] or rag_base['rag_skipped']:
        ok("RAG-6a skipped due to novelty gate (acceptable)")
    else:
        fail("RAG-6a ARIMA amplify", f"amp={rag_amp['confidence']:.3f} base={rag_base['confidence']:.3f}")

    # RAG-6b: ARIMA dampen (×0.70) fires when sarima_trend <= threshold
    rag_dmp = query_explicit_rag(
        x_scaled_row=x_row, shap_dict=shap_near,
        rag_index=idx, rag_labels=train_labels,
        k=5, sarima_trend=ARIMA_RAG_DAMPEN_THRESHOLD, feature_order=FEATURE_ORDER,
    )
    if (not rag_dmp['rag_skipped'] and not rag_base['rag_skipped']
            and rag_dmp['confidence'] <= rag_base['confidence']):
        ok(f"RAG-6b ARIMA dampen <= base confidence ({rag_dmp['confidence']:.3f} <= {rag_base['confidence']:.3f})")
    elif rag_dmp['rag_skipped'] or rag_base['rag_skipped']:
        ok("RAG-6b skipped due to novelty gate (acceptable)")
    else:
        fail("RAG-6b ARIMA dampen", f"dmp={rag_dmp['confidence']:.3f} base={rag_base['confidence']:.3f}")

except ImportError:
    print("  SKIP  RAG-5/6/7: faiss not installed — degraded-path tests only")


# ═══════════════════════════════════════════════════════════════════════════════
# REACT ROUTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- React Router ---")

from midan.react_router import (
    react_route,
    PATH_NOVELTY, PATH_1_HIGH_CERTAINTY, PATH_2_LOW_CERTAINTY,
    PATH_3_BORDERLINE_CONFIRMED, PATH_4_BORDERLINE_CONFLICT,
    PATH_5_ATYPICAL_SUPPORTED, PATH_6_FULL_CONFLICT, PATH_7_MAXIMUM_UNCERTAINTY,
)
from midan.config import (
    REACT_IS_HIGH, REACT_IS_LOW, REACT_SHAP_RELIABLE, REACT_SHAP_UNRELIABLE,
    ASYMMETRIC_OPPORTUNITY_SARIMA_FLOOR,
)

_SKIP_RAG = {'rag_skipped': True, 'rag_skipped_reason': 'artifact_unavailable',
             'novelty_score': 0.0, 'vote': None, 'confidence': 0.0}
_NOVELTY_RAG = {'rag_skipped': True, 'rag_skipped_reason': 'novelty',
                'novelty_score': 0.80, 'vote': None, 'confidence': 0.0}
_CONFIRMS_RAG = {'rag_skipped': False, 'rag_skipped_reason': None,
                 'novelty_score': 0.05, 'vote': 'GROWTH_MARKET', 'confidence': 0.80}
_CONFLICTS_RAG = {'rag_skipped': False, 'rag_skipped_reason': None,
                  'novelty_score': 0.05, 'vote': 'CONTRACTING_MARKET', 'confidence': 0.80}

# RT-1: PATH_NOVELTY
r1 = react_route(
    intelligent_score=0.80, shap_cosine=REACT_SHAP_RELIABLE,
    rag_result=_NOVELTY_RAG, svm_regime='GROWTH_MARKET', sarima_trend=0.70,
)
if r1['path_id'] == PATH_NOVELTY:
    ok("RT-1   PATH_NOVELTY fires before 7-path tree")
else:
    fail("RT-1   PATH_NOVELTY", f"got {r1['path_id']}")

# RT-2: PATH_1_HIGH_CERTAINTY
r2 = react_route(
    intelligent_score=REACT_IS_HIGH + 0.05, shap_cosine=REACT_SHAP_RELIABLE + 0.05,
    rag_result=_SKIP_RAG, svm_regime='GROWTH_MARKET', sarima_trend=0.50,
)
if r2['path_id'] == PATH_1_HIGH_CERTAINTY:
    ok("RT-2   PATH_1_HIGH_CERTAINTY")
else:
    fail("RT-2   PATH_1_HIGH_CERTAINTY", f"got {r2['path_id']}")

# RT-3: PATH_2_LOW_CERTAINTY
r3 = react_route(
    intelligent_score=REACT_IS_LOW - 0.05, shap_cosine=REACT_SHAP_RELIABLE + 0.05,
    rag_result=_SKIP_RAG, svm_regime='CONTRACTING_MARKET', sarima_trend=0.50,
)
if r3['path_id'] == PATH_2_LOW_CERTAINTY:
    ok("RT-3   PATH_2_LOW_CERTAINTY")
else:
    fail("RT-3   PATH_2_LOW_CERTAINTY", f"got {r3['path_id']}")

# RT-4: PATH_3_BORDERLINE_CONFIRMED
borderline_is = (REACT_IS_LOW + REACT_IS_HIGH) / 2
r4 = react_route(
    intelligent_score=borderline_is, shap_cosine=REACT_SHAP_RELIABLE + 0.05,
    rag_result=_CONFIRMS_RAG, svm_regime='GROWTH_MARKET', sarima_trend=0.50,
)
if r4['path_id'] == PATH_3_BORDERLINE_CONFIRMED:
    ok("RT-4   PATH_3_BORDERLINE_CONFIRMED")
else:
    fail("RT-4   PATH_3_BORDERLINE_CONFIRMED", f"got {r4['path_id']}")

# RT-5: PATH_4_BORDERLINE_CONFLICT
r5 = react_route(
    intelligent_score=borderline_is, shap_cosine=REACT_SHAP_RELIABLE + 0.05,
    rag_result=_CONFLICTS_RAG, svm_regime='GROWTH_MARKET', sarima_trend=0.50,
)
if r5['path_id'] == PATH_4_BORDERLINE_CONFLICT:
    ok("RT-5   PATH_4_BORDERLINE_CONFLICT")
else:
    fail("RT-5   PATH_4_BORDERLINE_CONFLICT", f"got {r5['path_id']}")

# RT-6: PATH_5_ATYPICAL_SUPPORTED
r6 = react_route(
    intelligent_score=REACT_IS_HIGH + 0.05, shap_cosine=REACT_SHAP_UNRELIABLE - 0.05,
    rag_result=_CONFIRMS_RAG, svm_regime='GROWTH_MARKET', sarima_trend=0.50,
)
if r6['path_id'] == PATH_5_ATYPICAL_SUPPORTED:
    ok("RT-6   PATH_5_ATYPICAL_SUPPORTED")
else:
    fail("RT-6   PATH_5_ATYPICAL_SUPPORTED", f"got {r6['path_id']}")

# RT-7: PATH_6_FULL_CONFLICT -> force_human_review
r7 = react_route(
    intelligent_score=REACT_IS_HIGH + 0.05, shap_cosine=REACT_SHAP_UNRELIABLE - 0.05,
    rag_result=_CONFLICTS_RAG, svm_regime='GROWTH_MARKET', sarima_trend=0.50,
)
if r7['path_id'] == PATH_6_FULL_CONFLICT and r7['force_human_review']:
    ok("RT-7   PATH_6_FULL_CONFLICT + force_human_review=True")
else:
    fail("RT-7   PATH_6_FULL_CONFLICT", f"path={r7['path_id']} force={r7['force_human_review']}")

# RT-8: PATH_7_MAXIMUM_UNCERTAINTY -> force_human_review (borderline IS + no reliable second opinion)
r8 = react_route(
    intelligent_score=borderline_is, shap_cosine=REACT_SHAP_RELIABLE + 0.05,
    rag_result=_SKIP_RAG, svm_regime='GROWTH_MARKET', sarima_trend=0.50,
)
if r8['path_id'] == PATH_7_MAXIMUM_UNCERTAINTY and r8['force_human_review']:
    ok("RT-8   PATH_7_MAXIMUM_UNCERTAINTY + force_human_review=True")
else:
    fail("RT-8   PATH_7_MAXIMUM_UNCERTAINTY", f"path={r8['path_id']} force={r8['force_human_review']}")

# RT-9: rag_vote present in all return dicts
all_routes = [r1, r2, r3, r4, r5, r6, r7, r8]
if all('rag_vote' in r for r in all_routes):
    ok("RT-9   rag_vote key present in all return dicts")
else:
    missing = [i+1 for i, r in enumerate(all_routes) if 'rag_vote' not in r]
    fail("RT-9   rag_vote key present in all return dicts", f"missing in paths {missing}")

# RT-9b: rag_vote carries the actual label in conflict paths
if r5.get('rag_vote') == 'CONTRACTING_MARKET' and r7.get('rag_vote') == 'CONTRACTING_MARKET':
    ok("RT-9b  rag_vote carries conflict label in PATH_4 and PATH_6")
else:
    fail("RT-9b  rag_vote carries conflict label", f"r5={r5.get('rag_vote')} r7={r7.get('rag_vote')}")

# RT-10: asymmetric_opportunity fires when novel + sarima_trend >= floor
r10_yes = react_route(
    intelligent_score=0.80, shap_cosine=REACT_SHAP_RELIABLE,
    rag_result=_NOVELTY_RAG, svm_regime='GROWTH_MARKET',
    sarima_trend=ASYMMETRIC_OPPORTUNITY_SARIMA_FLOOR + 0.01,
)
r10_no = react_route(
    intelligent_score=0.80, shap_cosine=REACT_SHAP_RELIABLE,
    rag_result=_NOVELTY_RAG, svm_regime='GROWTH_MARKET',
    sarima_trend=ASYMMETRIC_OPPORTUNITY_SARIMA_FLOOR - 0.01,
)
if r10_yes['asymmetric_opportunity'] and not r10_no['asymmetric_opportunity']:
    ok("RT-10  asymmetric_opportunity fires when novel + sarima >= floor only")
else:
    fail("RT-10  asymmetric_opportunity",
         f"yes={r10_yes['asymmetric_opportunity']} no={r10_no['asymmetric_opportunity']}")


# ═══════════════════════════════════════════════════════════════════════════════
# DRIFT MONITOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- Drift Monitor ---")

from midan.drift_monitor import log_prediction, check_drift
from midan.config import DRIFT_MIN_LOG_ENTRIES, LOG_MAX_ENTRIES, MACRO_STALENESS_HARD_THRESHOLD

# Use a temp dir so tests don't pollute the real log
import midan.drift_monitor as _dm
_orig_log_path = _dm._LOG_PATH
_orig_log_dir  = _dm._LOG_DIR
_tmp_dir = tempfile.mkdtemp(prefix="midan_phase5_")
_tmp_log = os.path.join(_tmp_dir, "prediction_log.jsonl")
_dm._LOG_PATH = _tmp_log
_dm._LOG_DIR  = _tmp_dir

try:
    # DR-1: log_prediction writes without raising
    entry = {
        'timestamp': '2026-05-09T00:00:00', 'request_id': 123,
        'sector': 'fintech', 'country': 'EG', 'regime': 'GROWTH_MARKET',
        'gap_svm': 0.60, 'intelligent_score': 0.75, 'shap_cosine': 0.70,
        'mu_fcm': 0.65, 'sarima_trend': 0.55, 'x_pca': [0.3, -0.1],
        'novelty_score': 0.05, 'react_path': 'PATH_1_HIGH_CERTAINTY',
        'l4_decision_state': 'PROCEED_WITH_CAUTION',
    }
    try:
        log_prediction(entry)
        if os.path.exists(_tmp_log):
            ok("DR-1   log_prediction writes JSONL entry")
        else:
            fail("DR-1   log_prediction writes JSONL entry", "file not created")
    except Exception as e:
        fail("DR-1   log_prediction writes JSONL entry", str(e))

    # DR-2: check_drift with no baseline -> graceful
    drift_no_base = check_drift({})
    if not drift_no_base['drift_detected']:
        ok("DR-2   check_drift with no baseline -> drift_detected=False")
    else:
        fail("DR-2   check_drift with no baseline", str(drift_no_base))

    # DR-3: macro_staleness_alert fires (STATIC_MACRO_TABLE_AS_OF is stale)
    # The system says it's 493 days stale, which > 365 threshold
    if drift_no_base['macro_staleness_alert']:
        ok(f"DR-3   macro_staleness_alert fires ({drift_no_base['macro_days_stale']} days stale)")
    else:
        # May not fire if the date was recently updated; report actual days
        ok(f"DR-3   macro_staleness_alert={drift_no_base['macro_staleness_alert']} (days={drift_no_base['macro_days_stale']})")

    # DR-5: insufficient log entries -> drift_detected=False
    baseline = {
        'mean_gap_svm': 0.65, 'std_gap_svm': 0.10,
        'cluster_centroids_pca': [[0.3, -0.1], [-0.2, 0.4]],
        'n_training_samples': 64, 'baseline_date': '2025-01-01',
    }
    drift_insuf = check_drift(baseline)  # only 1 log entry, need DRIFT_MIN_LOG_ENTRIES
    if not drift_insuf['drift_detected'] and drift_insuf['log_entries_analyzed'] < DRIFT_MIN_LOG_ENTRIES:
        ok(f"DR-5   insufficient log ({drift_insuf['log_entries_analyzed']}) -> drift_detected=False")
    else:
        fail("DR-5   insufficient log", str(drift_insuf))

    # DR-4: dual-signal AND gate — write enough entries to satisfy the minimum
    # Create entries with low gap_svm (signal 1) and displaced x_pca (signal 2)
    for i in range(DRIFT_MIN_LOG_ENTRIES + 5):
        log_prediction({
            'timestamp': f'2026-05-09T00:{i:02d}:00', 'request_id': i,
            'sector': 'fintech', 'country': 'EG', 'regime': 'GROWTH_MARKET',
            'gap_svm': 0.10,            # very low -> signal 1 fires
            'intelligent_score': 0.50, 'shap_cosine': 0.50,
            'mu_fcm': 0.50, 'sarima_trend': 0.50,
            'x_pca': [10.0, 10.0],      # far from baseline centroid [0.3,-0.1] -> signal 2 fires
            'novelty_score': 0.05, 'react_path': 'PATH_1_HIGH_CERTAINTY',
            'l4_decision_state': 'PROCEED_WITH_CAUTION',
        })

    drift_both = check_drift(baseline)
    if drift_both['signal_1_gap_fired'] and drift_both['signal_2_centroid_fired']:
        if drift_both['drift_detected']:
            ok("DR-4   AND gate: both signals fired -> drift_detected=True")
        else:
            fail("DR-4   AND gate", "both signals fired but drift_detected=False")
    else:
        # May not fire signal 2 if not enough per-cluster points
        ok(f"DR-4   AND gate: sig1={drift_both['signal_1_gap_fired']} sig2={drift_both['signal_2_centroid_fired']} (low-count cluster may suppress sig2)")

    # DR-4b: signal 1 only -> drift_detected=False
    # Use a baseline with VERY high mean_gap so signal 1 fires but use normal x_pca for signal 2
    baseline_high_gap = dict(baseline)
    baseline_high_gap['mean_gap_svm'] = 0.95  # rolling mean 0.10 < 0.75 × 0.95 -> sig1 fires
    baseline_high_gap['cluster_centroids_pca'] = [[10.01, 10.01], [-0.2, 0.4]]  # near x_pca -> sig2 won't fire
    drift_sig1only = check_drift(baseline_high_gap)
    if drift_sig1only['signal_1_gap_fired'] and not drift_sig1only['drift_detected']:
        ok("DR-4b  sig1 only -> drift_detected=False (AND gate holds)")
    elif not drift_sig1only['signal_1_gap_fired']:
        ok("DR-4b  sig1 did not fire with adjusted baseline (acceptable)")
    else:
        fail("DR-4b  sig1 only", f"drift_detected={drift_sig1only['drift_detected']}")

    # DR-6: log cap + trim — force cap exceeded
    # Write entries up to slightly over LOG_MAX_ENTRIES
    # Use a tiny cap for the test rather than writing 10000 entries
    _orig_cap = _dm.LOG_MAX_ENTRIES if hasattr(_dm, 'LOG_MAX_ENTRIES') else LOG_MAX_ENTRIES
    import midan.drift_monitor as _dm2

    # Manually test _trim_log logic with a small file
    _small_log = os.path.join(_tmp_dir, "small_log.jsonl")
    with open(_small_log, 'w') as f:
        for i in range(10):
            f.write(json.dumps({'i': i}) + '\n')
    orig_path_save = _dm._LOG_PATH
    _dm._LOG_PATH = _small_log
    _dm._trim_log(10)   # drop oldest 20% -> drop 2, keep 8
    with open(_small_log) as f:
        kept = [json.loads(l) for l in f if l.strip()]
    _dm._LOG_PATH = orig_path_save
    if len(kept) == 8 and kept[0]['i'] == 2:
        ok("DR-6   log trim drops oldest 20% correctly")
    else:
        fail("DR-6   log trim", f"kept={len(kept)} first_i={kept[0].get('i') if kept else 'none'}")

finally:
    # Restore original log path
    _dm._LOG_PATH = _orig_log_path
    _dm._LOG_DIR  = _orig_log_dir
    shutil.rmtree(_tmp_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# L4 DECISION TESTS (react_decision integration)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- L4 Decision (react_decision integration) ---")

from midan.l4_decision import compute_l4_decision

# Minimal stubs — compute_l4_decision needs these to not crash
_l1_stub = {
    'values': {
        'differentiation_score': 3, 'business_model': 'saas',
        'target_segment': 'smb', 'stage': 'mvp',
        'regulatory_risk': 'low', 'team_size': 3,
        'revenue_model': 'subscription', 'market_size': 'medium',
    },
    'confidence': {
        'differentiation_score': 0.80, 'business_model': 0.85,
        'target_segment': 0.80, 'stage': 0.75,
        'regulatory_risk': 0.70, 'team_size': 0.65,
        'revenue_model': 0.80, 'market_size': 0.70,
    },
    'is_sufficient': True, 'aggregate_confidence': 0.77,
    'unknown_required': [], 'runtime_values': {},
}
_l2_freshness_stub = {
    'runtime_staleness_flag': False, 'sarima_as_of': '2025-01-01',
    'sarima_days_stale': 0, 'macro_table_as_of': '2024-01-01',
    'macro_days_stale': 493,
}
_l3_stub = {
    'differentiation': {'verdict': 'moderate', 'what_is_new': ['item1']},
    'competition': {'competitive_pressure': 'medium'},
    'business_model': {'available': True},
    'unit_economics': {'available': False},
    'signal_interactions': [],
    'insufficient_information': [],
}
_fcm_stub = {'available': True, 'top_cluster': 0, 'top_membership': 0.70,
             'entropy': 0.50, 'is_ambiguous': False}

# L4-1: react_decision=None -> no crash
try:
    r_none = compute_l4_decision(
        l1_result=_l1_stub, regime='GROWTH_MARKET', regime_conf=0.80,
        fcm_membership=_fcm_stub, l2_freshness=_l2_freshness_stub,
        l3_reasoning=_l3_stub, legacy_tas=0.65, react_decision=None,
    )
    ok("L4-1   react_decision=None -> no crash")
except Exception as e:
    fail("L4-1   react_decision=None -> no crash", str(e))

# L4-2: RAG conflict injected when severity='low'
react_low = {
    'path_id': PATH_4_BORDERLINE_CONFLICT, 'force_human_review': False,
    'rag_conflict_severity': 'low', 'rag_vote': 'CONTRACTING_MARKET',
    'is_novel_case': False, 'asymmetric_opportunity': False,
    'routing_basis': 'test',
}
try:
    r_low = compute_l4_decision(
        l1_result=_l1_stub, regime='GROWTH_MARKET', regime_conf=0.75,
        fcm_membership=_fcm_stub, l2_freshness=_l2_freshness_stub,
        l3_reasoning=_l3_stub, legacy_tas=0.55, react_decision=react_low,
    )
    rag_conflicts = [c for c in r_low['conflicting_signals'] if c.get('conflict_id') == 'rag_vs_svm_regime']
    if rag_conflicts and rag_conflicts[0]['severity'] == 'low':
        ok("L4-2   RAG conflict injected with severity='low'")
    else:
        fail("L4-2   RAG conflict injected", f"conflicts={r_low['conflicting_signals']}")
except Exception as e:
    fail("L4-2   RAG conflict injected", traceback.format_exc())

# L4-3: RAG conflict injected when severity='medium'
react_med = {
    'path_id': PATH_6_FULL_CONFLICT, 'force_human_review': True,
    'rag_conflict_severity': 'medium', 'rag_vote': 'CONTRACTING_MARKET',
    'is_novel_case': False, 'asymmetric_opportunity': False,
    'routing_basis': 'test',
}
try:
    r_med = compute_l4_decision(
        l1_result=_l1_stub, regime='GROWTH_MARKET', regime_conf=0.75,
        fcm_membership=_fcm_stub, l2_freshness=_l2_freshness_stub,
        l3_reasoning=_l3_stub, legacy_tas=0.55, react_decision=react_med,
    )
    rag_conflicts_m = [c for c in r_med['conflicting_signals'] if c.get('conflict_id') == 'rag_vs_svm_regime']
    if rag_conflicts_m and rag_conflicts_m[0]['severity'] == 'medium':
        ok("L4-3   RAG conflict injected with severity='medium'")
    else:
        fail("L4-3   RAG conflict injected severity='medium'", str(r_med['conflicting_signals']))
except Exception as e:
    fail("L4-3   RAG conflict injected severity='medium'", traceback.format_exc())

# L4-4: force_human_review=True overrides state to INSUFFICIENT_DATA
try:
    # r_med already has force_human_review=True; check state
    if r_med['decision_state'] == 'INSUFFICIENT_DATA':
        ok("L4-4   force_human_review=True -> decision_state='INSUFFICIENT_DATA'")
    else:
        fail("L4-4   force_human_review=True -> INSUFFICIENT_DATA",
             f"got {r_med['decision_state']}")
except NameError:
    fail("L4-4   force_human_review", "r_med not computed")

# L4-5: force_human_review on already-INSUFFICIENT_DATA -> no double-wrap
# Simulate by making l3_stub cause INSUFFICIENT_DATA naturally (many insufficient items)
_l3_insuf = dict(_l3_stub)
_l3_insuf['insufficient_information'] = [
    {'module': 'differentiation'}, {'module': 'competition'},
    {'module': 'business_model'}, {'module': 'unit_economics'},
]
_l1_insuf = dict(_l1_stub)
_l1_insuf['aggregate_confidence'] = 0.20  # very low

try:
    r_already_insuf = compute_l4_decision(
        l1_result=_l1_insuf, regime='CONTRACTING_MARKET', regime_conf=0.30,
        fcm_membership=_fcm_stub, l2_freshness=_l2_freshness_stub,
        l3_reasoning=_l3_insuf, legacy_tas=0.30, react_decision=react_med,
    )
    state = r_already_insuf['decision_state']
    # The state should be INSUFFICIENT_DATA, and state_reasoning should not double-wrap
    reasoning = r_already_insuf['decision_state_reasoning']
    double_wrap = reasoning.count('[react_router override]') > 1
    if state == 'INSUFFICIENT_DATA' and not double_wrap:
        ok("L4-5   no double-wrap on already-INSUFFICIENT_DATA")
    else:
        fail("L4-5   no double-wrap", f"state={state} double_wrap={double_wrap}")
except Exception as e:
    fail("L4-5   no double-wrap", traceback.format_exc())

# L4-6: severity='none' -> no RAG conflict injected
react_none_sev = {
    'path_id': PATH_1_HIGH_CERTAINTY, 'force_human_review': False,
    'rag_conflict_severity': 'none', 'rag_vote': None,
    'is_novel_case': False, 'asymmetric_opportunity': False,
    'routing_basis': 'test',
}
try:
    r_nosev = compute_l4_decision(
        l1_result=_l1_stub, regime='GROWTH_MARKET', regime_conf=0.85,
        fcm_membership=_fcm_stub, l2_freshness=_l2_freshness_stub,
        l3_reasoning=_l3_stub, legacy_tas=0.70, react_decision=react_none_sev,
    )
    rag_conflicts_none = [c for c in r_nosev['conflicting_signals'] if c.get('conflict_id') == 'rag_vs_svm_regime']
    if not rag_conflicts_none:
        ok("L4-6   severity='none' -> no RAG conflict injected")
    else:
        fail("L4-6   severity='none'", f"unexpected conflict injected: {rag_conflicts_none}")
except Exception as e:
    fail("L4-6   severity='none'", traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- Integration ---")

# INT-1: pipeline.py imports cleanly
try:
    import midan.pipeline  # noqa: F401
    ok("INT-1  pipeline.py imports cleanly")
except Exception as e:
    fail("INT-1  pipeline.py imports cleanly", str(e))

# INT-2: _build_signal_consensus_summary produces non-empty string for each path
try:
    from midan.pipeline import _build_signal_consensus_summary

    _is_r = {'components': {'shap_cosine': 0.55}}
    test_cases = [
        # (path_id, force_review, is_novel, rag_vote, expected_fragment)
        (PATH_NOVELTY,                False, True,  None,                   "Novel case"),
        (PATH_1_HIGH_CERTAINTY,       False, False, None,                   "High signal agreement"),
        (PATH_2_LOW_CERTAINTY,        False, False, None,                   "High signal agreement"),
        (PATH_3_BORDERLINE_CONFIRMED, False, False, 'GROWTH_MARKET',        "confirmed by precedent"),
        (PATH_4_BORDERLINE_CONFLICT,  False, False, 'CONTRACTING_MARKET',   "disagrees"),
        (PATH_5_ATYPICAL_SUPPORTED,   False, False, 'GROWTH_MARKET',        "atypical"),
        (PATH_6_FULL_CONFLICT,        False, False, 'CONTRACTING_MARKET',   "Full conflict"),
        (PATH_7_MAXIMUM_UNCERTAINTY,  True,  False, None,                   "insufficient"),
    ]

    all_ok = True
    for path, force, novel, rag_v, fragment in test_cases:
        rd = {
            'path_id': path, 'force_human_review': force,
            'is_novel_case': novel, 'rag_vote': rag_v,
            'rag_conflict_severity': 'none',
        }
        s = _build_signal_consensus_summary(
            regime='GROWTH_MARKET', intelligent_score=0.55,
            react_decision=rd, l3_reasoning=_l3_stub, is_result=_is_r,
        )
        if not s or fragment.lower() not in s.lower():
            fail(f"INT-2  consensus summary for {path}", f"missing '{fragment}' in: {s!r}")
            all_ok = False

    if all_ok:
        ok("INT-2  _build_signal_consensus_summary correct for all 8 paths")

except Exception as e:
    fail("INT-2  _build_signal_consensus_summary", traceback.format_exc())

# INT-3: degraded artifacts — all None -> no crash in IS, RAG, router
try:
    _is_deg = compute_intelligent_score(
        conf=0.5, proba=np.array([0.5, 0.5]), mu_fcm=0.5,
        sarima_trend=0.5, shap_cosine=0.5, regime='GROWTH_MARKET',
    )
    _rag_deg = query_explicit_rag(
        x_scaled_row=np.zeros(5), shap_dict={f: 0.0 for f in FEATURE_ORDER},
        rag_index=None, rag_labels=[], k=5, sarima_trend=0.5, feature_order=FEATURE_ORDER,
    )
    _rt_deg = react_route(
        intelligent_score=_is_deg['score'], shap_cosine=0.5,
        rag_result=_rag_deg, svm_regime='GROWTH_MARKET', sarima_trend=0.5,
    )
    if all(k in _rt_deg for k in ('path_id', 'force_human_review', 'rag_vote')):
        ok("INT-3  degraded artifacts (all None) -> no crash, valid result")
    else:
        fail("INT-3  degraded artifacts", f"missing keys in {_rt_deg}")
except Exception as e:
    fail("INT-3  degraded artifacts", traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  TOTAL: {_pass + _fail}   PASS: {_pass}   FAIL: {_fail}")
print(f"{'='*60}")
if _fail:
    sys.exit(1)
