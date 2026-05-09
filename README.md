# MIDAN — Market Intelligence & Decision Analysis Network

An epistemically disciplined, multi-layer AI pipeline that evaluates startup ideas against macro market signals, extracts structural competitive mechanisms, and delivers calibrated, traceable strategic verdicts.

Every output field is grounded in a specific signal path. Every claim is bounded by the evidence quality that generated it.

---

## Architecture

```
Founder Input (natural language)
         |
         v
[L0] Sanity Gate
     3-tier rejection: IMPOSSIBLE / BROKEN / INCOMPLETE
     Hard stop before any ML runs
         |
         v
[L1] Confidence-Scored Feature Extraction
     8 structured fields + per-field confidence + UNKNOWN sentinels
     Sufficiency gate: blocks inference when required fields are UNKNOWN
         |
         v
[L2A]   Static macro vector — (sector, country) lookup
[L2A.5] Idea-perturbed macro — L1-gated traceable deltas
[L2B]   SVM RBF + rule overrides — regime classification
[L2B.5] FCM fuzzy membership — top cluster, membership score, ambiguity
[L2C]   LightGBM + SHAP — macro signal explainability
[L2C.5] Implicit RAG — shap_cosine: attribution consistency vs cluster mean
[L2D]   SARIMA precomputed table — 90-day sector trend
[L2D.5] Intelligent Score (IS) — 5-signal routing composite
[L2E]   Freshness gate — staleness penalty when SARIMA is stale
         |
         v
[L3] Structured Reasoning
     differentiation / competition / business_model / unit_economics
     signal_interactions — cross-field amplifications and conflicts
     insufficient_information — explicit when signal coverage is thin
         |
         v
[MECHANISM] 8-Phase Extraction Pipeline
     Extracts up to 13 structural competitive mechanisms
     Epistemic inference ladder: observed / inferred / speculative
     Uncertainty value feeds L4 as a continuous probabilistic modifier
         |
         v
[ReAct] Explicit RAG + Routing
     Novelty gate: 1 - max_cosine_k_neighbors (suppresses RAG on novel inputs)
     FAISS k-NN over ~64 macro grid training points -> majority vote
     ARIMA modifier: amplifies/dampens confidence based on sector trend
     react_route -> one of 8 deterministic named paths
     signal_consensus_summary -> resolves contradictions before explanation layer
         |
         v
[L4] Decision State Machine
     Risk decomposition: market / execution / timing (independent dimensions)
     Conflict detection: severity-tiered, includes RAG conflict injection
     Offsetting analysis: strong signals can offset high risk
     force_human_review override -> INSUFFICIENT_DATA when router escalates
     decision_state: PROCEED / VALIDATE / REVISIT / HIGH_UNCERTAINTY /
                     CONFLICTING_SIGNALS / INSUFFICIENT_DATA
         |
         v
[Synthesis] Epistemically Calibrated Output
     7 structured strategic fields with per-field signal anchors
     Language tone conditional on evidence_quality (strong/moderate/weak/insufficient)
     epistemic_disclosure woven into prose
         |
         v
[Log] Prediction log (JSONL) — feeds drift detection
```

---

## Intelligent Score (IS)

IS is a **routing signal only** — it determines which reasoning path the system takes. It never sets `decision_state` directly. L4 remains the decision authority.

| Signal | Weight | Source |
|---|---|---|
| S — regime favorability | 0.20 | Fixed lookup per regime label |
| gap_svm — probability margin | 0.25 | `proba_sorted[0] - proba_sorted[1]` (NOT `proba.max()`) |
| mu_fcm — top cluster membership | 0.20 | FCM soft assignment |
| arima — sector trend | 0.15 | SARIMA normalized forecast |
| shap_cosine — attribution consistency | 0.20 | Implicit RAG cosine similarity |

**Correlated-trio discount:** `gap_svm`, `mu_fcm`, and `shap_cosine` all derive from the same `x_scaled` geometry. When all three exceed 0.80 simultaneously, their combined contribution is multiplied by 0.85 to prevent fake consensus inflation.

---

## RAG Layer

### Implicit RAG — `shap_cosine` (L2C.5)

Measures whether the model is reasoning about the current input the same way it learned to reason about similar inputs in its cluster. Computed as `cosine(current_SHAP_vector, cluster_mean_SHAP_vector)`.

- `1.0` — attribution pattern matches the cluster's typical pattern
- `0.5` — neutral default when the artifact is absent (never penalizes)
- `0.0` — attribution completely orthogonal to the cluster mean

This is an **attribution consistency signal**, not a prediction validity signal.

### Explicit RAG — FAISS k-NN (ReAct block)

Retrieves the k=5 most structurally similar training grid points and takes a majority vote on their regime labels. The query vector is a 10D combination of `[x_scaled (5D), shap_shares (5D)]`, L2-normalized.

**Novelty gate:** if `1 - max_cosine_among_k_neighbors > 0.40`, RAG is suppressed entirely. A vote from structurally distant neighbors is noise, not evidence.

**ARIMA modifier:** sector trend amplifies (`+0.10`) or dampens (`×0.70`) confidence in historical precedent based on current direction.

**Authority:** RAG is advisory at every level. It feeds the router; the router is advisory to L4.

---

## ReAct Routing Paths

| Path | Condition | force_human_review |
|---|---|---|
| PATH_NOVELTY | `novelty_score > 0.40` — no structural precedent | No |
| PATH_1_HIGH_CERTAINTY | IS >= 0.72, shap reliable, no RAG conflict | No |
| PATH_2_LOW_CERTAINTY | IS <= 0.35, shap reliable | No |
| PATH_3_BORDERLINE_CONFIRMED | IS borderline, shap reliable, RAG confirms SVM | No |
| PATH_4_BORDERLINE_CONFLICT | IS borderline, shap reliable, RAG conflicts | No |
| PATH_5_ATYPICAL_SUPPORTED | shap atypical, RAG confirms SVM | No |
| PATH_6_FULL_CONFLICT | shap atypical AND RAG conflicts — two doubt sources | **Yes** |
| PATH_7_MAXIMUM_UNCERTAINTY | catch-all — no reliable second opinion | **Yes** |

---

## Mechanism Extraction

Runs between L3 and L4. Extracts structural competitive mechanisms and feeds uncertainty back into L4 probabilistically.

**Mechanism types:** `network_effect`, `switching_cost`, `brand_moat`, `data_moat`, `regulatory_moat`, `cost_advantage`, `process_efficiency`, `distribution_control`, `api_dependency`, `platform_dependency`, `regulatory_headwind`

**Epistemic inference ladder:**

| Tier | `inference_depth` | Max claim level |
|---|---|---|
| 3 | `directly_observed` | `strategic_conclusion` |
| 2 | `one_step_inference` | `inference` |
| 1 | `speculative` | `observation` only |

---

## Drift Monitoring

`midan/drift_monitor.py` maintains an append-only prediction log (`logs/prediction_log.jsonl`) and implements a dual-signal AND-gate drift detector.

**Drift requires both signals to fire simultaneously:**
- Signal 1 — rolling mean `gap_svm` < 0.75 × baseline mean (classification margin declining)
- Signal 2 — FCM centroid displacement >= 2 clusters moved > 1.5 L2 distance

A single signal firing is noise. Both firing simultaneously is confirmed drift.

**Macro staleness alert** fires independently when `STATIC_MACRO_TABLE_AS_OF` exceeds 365 days — no prediction log required.

---

## Market Regimes

| Regime | Description |
|---|---|
| `GROWTH_MARKET` | Expanding deal volume, high sector momentum |
| `EMERGING_MARKET` | Early-stage activity, increasing signal density |
| `HIGH_FRICTION_MARKET` | Active but structurally constrained |
| `CONTRACTING_MARKET` | Declining volume, adverse macro conditions |

---

## Decision States

| State | Meaning |
|---|---|
| `PROCEED` | Risk decomposition supports forward movement |
| `VALIDATE` | Promising but a specific assumption must be tested first |
| `REVISIT` | Structural issues need addressing before proceeding |
| `HIGH_UNCERTAINTY` | Signals too divergent for confident recommendation |
| `CONFLICTING_SIGNALS` | Multiple independent doubt sources active |
| `INSUFFICIENT_DATA` | Signal coverage too thin to evaluate, or router escalated |

---

## Setup

### Prerequisites
- Python 3.10+
- Groq API key (free tier at [console.groq.com](https://console.groq.com))

### Installation

```bash
git clone https://github.com/seifalaa111/Intelligent_Systems_Project.git
cd Intelligent_Systems_Project

pip install -r requirements.txt

# Generate model artifacts
# Open MIDAN_Pipeline.ipynb in Jupyter or Google Colab and run all cells.
# This produces .pkl files in models/ (gitignored due to size).
# Cells T1-T4 generate the ReAct/RAG artifacts:
#   shap_cluster_means.pkl, rag_index.pkl, rag_labels.json,
#   drift_baseline.json, regime_anchors_pca.pkl

# Configure environment
# Create .env with:
#   GROQ_API_KEY=your_key_here
#   SLACK_WEBHOOK_URL=optional
```

---

## Running

```bash
# FastAPI backend + HTML frontend
uvicorn api:api --host 0.0.0.0 --port 8000 --reload
# Open midan.html in a browser

# Streamlit dashboard
streamlit run app.py

# Both simultaneously (Windows)
.\start_servers.ps1

# Test suite
pytest test_pipeline_logic.py test_api.py test_interact_logic.py test_response_schema.py -v

# Phase 5 validation (ReAct/RAG components)
python validate_phase5.py
```

---

## API Reference

### `POST /process`

```json
{
  "idea_text": "A B2B SaaS for Egyptian logistics companies to optimize last-mile routing",
  "sector": "logistics",
  "country": "EG"
}
```

Key response fields:

```json
{
  "decision_state": "VALIDATE",
  "decision_strength": { "tier": "moderate" },
  "risk_decomposition": {
    "market_risk":    { "level": "moderate", "driver": "..." },
    "execution_risk": { "level": "high",     "driver": "..." },
    "timing_risk":    { "level": "low",      "driver": "..." }
  },
  "intelligent_score": 0.61,
  "react_path": "PATH_3_BORDERLINE_CONFIRMED",
  "signal_consensus": "Borderline IS=0.61 confirmed by precedent analysis...",
  "rag_result": { "vote": "EMERGING_MARKET", "confidence": 0.72, "novelty_score": 0.08 },
  "react_decision": { "path_id": "PATH_3_BORDERLINE_CONFIRMED", "force_human_review": false },
  "conflicting_signals": [],
  "strategic_interpretation": "...",
  "key_driver": "...",
  "main_risk": "...",
  "counterpoint": "...",
  "action": "...",
  "mechanism_analysis": {
    "extraction_mode": "partial",
    "uncertainty": 0.14,
    "epistemic_summary": { "evidence_quality": "moderate", "recommended_disclosure": "..." }
  },
  "l2_data_freshness": { "runtime_staleness_flag": false },
  "fcm_membership": { "top_cluster": 2, "top_membership": 0.74, "is_ambiguous": false }
}
```

### `POST /chat`
Conversational follow-up. Reads only the L4 decision envelope — never speculates beyond what the pipeline concluded.

### `GET /health`
Pipeline status and model availability.

---

## Project Structure

```
Intelligent_Systems_Project/
|
+-- api.py                        # FastAPI app — /process, /chat, /health
+-- app.py                        # Streamlit dashboard
+-- midan.html                    # Static HTML/JS frontend
+-- MIDAN_Pipeline.ipynb          # Training notebook — generates all model artifacts
+-- validate_phase5.py            # Phase 5 validation suite (39 tests)
+-- requirements.txt
|
+-- midan/
|   +-- config.py                 # All tunable constants (IS weights, routing thresholds, etc.)
|   +-- core.py                   # Artifact loading, shared utilities, ResponsePayload schema
|   +-- pipeline.py               # Main orchestrator — L0 -> L1 -> L2 -> L3 -> Mech -> ReAct -> L4
|   +-- l0_gate.py                # L0: 3-tier sanity gate
|   +-- l1_parser.py              # L1: confidence-scored feature extraction
|   +-- l2_intelligence.py        # L2: SVM, FCM, SHAP, SARIMA, regime classification
|   +-- l3_reasoning.py           # L3: structured reasoning modules
|   +-- intelligent_score.py      # IS: 5-signal routing composite with trio discount
|   +-- rag.py                    # Implicit RAG (shap_cosine) + Explicit RAG (FAISS k-NN)
|   +-- react_router.py           # Deterministic 8-path routing tree
|   +-- l4_decision.py            # L4: decision state machine + risk decomposition
|   +-- mechanism_extractor.py    # 8-phase mechanism pipeline + epistemic summary
|   +-- drift_monitor.py          # Prediction log + dual-signal drift detection
|   +-- outer_react.py            # Retrain loop: cluster remapping + SHAP drift gate
|   +-- response.py               # Synthesis layer + epistemic calibration
|   +-- conversation.py           # Chat routing + post-decision mode branching
|   +-- endpoints.py              # FastAPI route handlers
|
+-- models/                       # Artifact directory
|   +-- *.pkl                     # Binary model files — gitignored (regenerate via notebook)
|   +-- sarima_results.json       # Pre-computed SARIMA forecasts
|   +-- competitors_context.json
|   +-- sentiment_context.json
|   +-- shap_feature_importance.json
|   # ReAct/RAG artifacts (generated by T1-T4 in notebook, gitignored):
|   # shap_cluster_means.pkl, rag_index.pkl, rag_labels.json,
|   # drift_baseline.json, regime_anchors_pca.pkl
|
+-- Datasets/                     # Raw training datasets
+-- PROJECT_SCRAPPED_DATA/        # Intelligence data collection subsystem
+-- logs/                         # Runtime prediction log (gitignored)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Frontend | Vanilla HTML/CSS/JS |
| ML Models | scikit-learn, LightGBM, statsmodels |
| Fuzzy Logic | scikit-fuzzy (FCM) |
| Vector Search | FAISS (explicit RAG index) |
| Explainability | SHAP |
| LLM Inference | Groq (Llama 3.1 8B Instant) |

---

## Coverage

**Sectors:** Fintech, E-commerce, Healthtech, Edtech, SaaS, Logistics, Agritech

**Countries:** Egypt, Saudi Arabia, UAE, Morocco, Nigeria, Kenya, United States, United Kingdom

---

## Design Principles

- **Epistemic discipline** — every synthesis claim is bounded by the evidence quality that generated it. Observed, inferred, and speculative conclusions are explicitly distinguished.
- **Signal authority hierarchy** — IS is a routing signal; the router is advisory to L4; L4 is the sole decision authority. No component can bypass this chain.
- **Fail-soft on missing artifacts** — all new ML artifacts (SHAP cluster means, FAISS index, drift baseline) load as `None` on absence and return safe neutral defaults. The system degrades gracefully, never crashes.
- **Training source integrity** — `outer_react.py` enforces a hard contract: prediction logs never feed retraining. Only macro grid tables are training sources.
- **Full traceability** — every L4 text field is annotated with the L1/L2/L3 signal paths that justify it via `strategic_anchors`. Every routing decision carries a `routing_basis` string.
