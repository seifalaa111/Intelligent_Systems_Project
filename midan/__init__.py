"""
midan — MIDAN AI Decision Engine, modular package.

Layer modules:
  core              — shared schemas, constants, ML model loaders, utilities
  l0_gate           — L0 input validation gate (length, contradictions, spam, vague, …)
  l1_parser         — confidence-scored idea feature extraction + consistency
  l2_intelligence   — macro/regime/FCM/SHAP/SARIMA + freshness, idea-perturbed adjustments
  l3_reasoning      — structured idea reasoning (differentiation, competition, BM, unit economics, interactions)
  l4_decision       — risk decomposition, conflict detection, offsetting, decision state machine
  conversation      — intent classification, post-decision routing, follow-up helpers
  response          — payload builder, chat fallback, operator reply, projection, L4 strategic generators
  pipeline          — process_idea + run_inference orchestrators
  endpoints         — FastAPI app + HTTP endpoints

The legacy `api.py` module remains as a re-export shim — `import api` still
works and resolves all symbols against this package.
"""
