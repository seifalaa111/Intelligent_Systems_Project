# MIDAN — Market Intelligence & Decision Analysis Network

> An AI pipeline that evaluates startup ideas against real macro market signals and delivers **calibrated, traceable strategic verdicts** — not opinions.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?logo=fastapi&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-green)
![FAISS](https://img.shields.io/badge/FAISS-RAG-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What Is MIDAN?

MIDAN is a multi-layer ML pipeline that takes a startup idea described in plain English and returns a structured strategic decision — backed by market regime classification, fuzzy cluster analysis, SHAP explainability, SARIMA trend forecasting, RAG-based precedent retrieval, and mechanism extraction.

Every output field is grounded in a specific signal path. Every claim is bounded by the evidence quality that generated it. The system cannot produce a confident recommendation when the evidence does not support one.

**Supported sectors:** Fintech · E-commerce · Healthtech · Edtech · SaaS · Logistics · Agritech  
**Supported countries:** Egypt · Saudi Arabia · UAE · Morocco · Nigeria · Kenya · United States · United Kingdom

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Groq API key](https://console.groq.com) (free tier)

### Installation

```bash
git clone https://github.com/seifalaa111/Intelligent_Systems_Project.git
cd Intelligent_Systems_Project

pip install -r requirements.txt
pip install faiss-cpu  # for RAG vector search
```

### Generate Model Artifacts

Open `MIDAN_Pipeline.ipynb` in Jupyter or Google Colab and run all cells. This produces `.pkl` files in `models/` (gitignored due to size). Cells T1–T4 generate the ReAct/RAG artifacts:

```
models/shap_cluster_means.pkl
models/rag_index.pkl
models/rag_labels.json
models/drift_baseline.json
models/regime_anchors_pca.pkl
```

### Configure Environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
SLACK_WEBHOOK_URL=optional_for_drift_alerts
```

### Run

```bash
# FastAPI backend — then open midan.html in a browser
uvicorn api:api --host 0.0.0.0 --port 8000 --reload

# Streamlit dashboard
streamlit run app.py

# Both simultaneously (Windows)
.\start_servers.ps1
```

### Test

```bash
pytest test_pipeline_logic.py test_api.py test_interact_logic.py test_response_schema.py -v

# Phase 5 ReAct/RAG validation
python validate_phase5.py
```

---

## Architecture

Every idea is processed through a strict, ordered pipeline. Each stage either enriches the signal or raises a hard stop.

```
Founder Input (natural language)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  L0  Sanity Gate                                        │
│  3-tier rejection: IMPOSSIBLE / BROKEN / INCOMPLETE     │
│  Hard stop — no ML runs on nonsense inputs              │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  L1  Feature Extraction                                 │
│  8 structured fields · per-field confidence scores      │
│  UNKNOWN sentinels · sufficiency gate                   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  L2  Market Intelligence                                │
│  L2A    Static macro vector (sector × country lookup)  │
│  L2A.5  Idea-perturbed macro (L1-gated deltas)         │
│  L2B    SVM RBF + rule overrides → regime label        │
│  L2B.5  FCM fuzzy membership → cluster + ambiguity     │
│  L2C    LightGBM + SHAP → macro explainability         │
│  L2C.5  Implicit RAG → shap_cosine attribution check  │
│  L2D    SARIMA precomputed table → 90-day trend        │
│  L2D.5  Intelligent Score (IS) — 5-signal composite    │
│  L2E    Freshness gate → staleness penalty             │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  L3  Structured Reasoning                               │
│  Differentiation · Competition · Business Model         │
│  Unit Economics · Signal Interactions                   │
│  Explicit insufficient_information state                │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Mechanism Extractor  (8-phase pipeline)                │
│  Up to 13 structural competitive mechanisms             │
│  Epistemic ladder: observed / inferred / speculative    │
│  Uncertainty score → L4 probabilistic modifier         │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  ReAct Layer                                            │
│  Novelty gate: suppresses RAG on structurally novel     │
│  FAISS k-NN (k=5) over macro grid → majority vote      │
│  ARIMA modifier: amplifies/dampens historical confidence│
│  8 deterministic named routing paths                    │
│  Signal consensus summary → resolves contradictions     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  L4  Decision State Machine                             │
│  Risk decomposition: market / execution / timing        │
│  Conflict detection (severity-tiered)                   │
│  Offsetting analysis: strong signals can offset risk    │
│  force_human_review on router escalation                │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Synthesis  Epistemically Calibrated Output             │
│  7 structured strategic fields with signal anchors      │
│  Tone conditioned on evidence_quality                   │
│  Prediction logged → feeds drift detection              │
└─────────────────────────────────────────────────────────┘
```

---

## Decision States

| State | Meaning |
|---|---|
| `PROCEED` | Risk decomposition supports moving forward |
| `VALIDATE` | Promising — a specific assumption must be tested first |
| `REVISIT` | Structural issues need addressing before proceeding |
| `HIGH_UNCERTAINTY` | Signals too divergent for a confident recommendation |
| `CONFLICTING_SIGNALS` | Multiple independent doubt sources active simultaneously |
| `INSUFFICIENT_DATA` | Signal coverage too thin, or router escalated to human review |

---

## Market Regimes

| Regime | Description |
|---|---|
| `GROWTH_MARKET` | Expanding deal volume, high sector momentum |
| `EMERGING_MARKET` | Early-stage activity, increasing signal density |
| `HIGH_FRICTION_MARKET` | Active but structurally constrained |
| `CONTRACTING_MARKET` | Declining volume, adverse macro conditions |

---

## Intelligent Score (IS)

IS is a **routing signal only** — it determines which reasoning path the system takes. It never directly sets `decision_state`. L4 is the sole decision authority.

| Signal | Weight | Source |
|---|---|---|
| S — regime favorability | 0.20 | Fixed lookup per regime label |
| gap_svm — probability margin | 0.25 | `proba_sorted[0] - proba_sorted[1]` |
| mu_fcm — top cluster membership | 0.20 | FCM soft assignment |
| arima — sector trend | 0.15 | SARIMA normalized forecast |
| shap_cosine — attribution consistency | 0.20 | Implicit RAG cosine similarity |

**Correlated-trio discount:** when `gap_svm`, `mu_fcm`, and `shap_cosine` all exceed 0.80 simultaneously, their combined contribution is multiplied by 0.85 to prevent inflated false consensus.

---

## ReAct Routing Paths

| Path | Trigger | Human Review? |
|---|---|---|
| `PATH_NOVELTY` | `novelty_score > 0.40` — no structural precedent | No |
| `PATH_1_HIGH_CERTAINTY` | IS ≥ 0.72, SHAP reliable, no RAG conflict | No |
| `PATH_2_LOW_CERTAINTY` | IS ≤ 0.35, SHAP reliable | No |
| `PATH_3_BORDERLINE_CONFIRMED` | IS borderline, SHAP reliable, RAG confirms SVM | No |
| `PATH_4_BORDERLINE_CONFLICT` | IS borderline, SHAP reliable, RAG conflicts | No |
| `PATH_5_ATYPICAL_SUPPORTED` | SHAP atypical, RAG confirms SVM | No |
| `PATH_6_FULL_CONFLICT` | SHAP atypical AND RAG conflicts — two independent doubt sources | **Yes** |
| `PATH_7_MAXIMUM_UNCERTAINTY` | Catch-all — no reliable second opinion | **Yes** |

---

## API Reference

### `POST /process`

Evaluate a startup idea end-to-end.

**Request:**
```json
{
  "idea_text": "A B2B SaaS for Egyptian logistics companies to optimize last-mile routing",
  "sector": "logistics",
  "country": "EG"
}
```

**Key response fields:**
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
  "rag_result": {
    "vote": "EMERGING_MARKET",
    "confidence": 0.72,
    "novelty_score": 0.08
  },
  "mechanism_analysis": {
    "extraction_mode": "partial",
    "uncertainty": 0.14,
    "epistemic_summary": {
      "evidence_quality": "moderate",
      "recommended_disclosure": "..."
    }
  },
  "strategic_interpretation": "...",
  "key_driver": "...",
  "main_risk": "...",
  "counterpoint": "...",
  "action": "..."
}
```

### `POST /chat`

Conversational follow-up anchored to the L4 decision envelope. Never speculates beyond what the pipeline concluded.

### `GET /health`

Pipeline status and model artifact availability.

---

## Project Structure

```
Intelligent_Systems_Project/
│
├── api.py                        # FastAPI app — /process, /chat, /health
├── app.py                        # Streamlit dashboard
├── midan.html                    # Static HTML/JS frontend
├── MIDAN_Pipeline.ipynb          # Training notebook — generates all model artifacts
├── validate_phase5.py            # ReAct/RAG validation suite
├── requirements.txt
│
├── midan/
│   ├── config.py                 # Tunable constants (IS weights, routing thresholds)
│   ├── core.py                   # Artifact loading, shared utilities, ResponsePayload schema
│   ├── pipeline.py               # Main orchestrator — L0 → L1 → L2 → L3 → Mech → ReAct → L4
│   ├── l0_gate.py                # L0: 3-tier sanity gate
│   ├── l1_parser.py              # L1: confidence-scored feature extraction
│   ├── l2_intelligence.py        # L2: SVM, FCM, SHAP, SARIMA, regime classification
│   ├── l3_reasoning.py           # L3: structured reasoning modules
│   ├── intelligent_score.py      # IS: 5-signal routing composite with trio discount
│   ├── rag.py                    # Implicit RAG (shap_cosine) + Explicit RAG (FAISS k-NN)
│   ├── react_router.py           # Deterministic 8-path routing tree
│   ├── l4_decision.py            # L4: decision state machine + risk decomposition
│   ├── mechanism_extractor.py    # 8-phase mechanism pipeline + epistemic summary
│   ├── drift_monitor.py          # Prediction log + dual-signal drift detection
│   ├── outer_react.py            # Retrain loop: cluster remapping + SHAP drift gate
│   ├── response.py               # Synthesis layer + epistemic calibration
│   ├── conversation.py           # Chat routing + post-decision mode branching
│   └── endpoints.py              # FastAPI route handlers
│
├── models/                       # Artifact directory (gitignored — regenerate via notebook)
│   ├── sarima_results.json
│   ├── competitors_context.json
│   ├── sentiment_context.json
│   └── shap_feature_importance.json
│
├── Datasets/                     # Raw training datasets
├── PROJECT_SCRAPPED_DATA/        # Intelligence data collection subsystem
└── logs/                         # Runtime prediction log (gitignored)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Frontend | Vanilla HTML / CSS / JS |
| ML Classification | scikit-learn (SVM RBF), LightGBM |
| Fuzzy Clustering | scikit-fuzzy (FCM) |
| Time Series | statsmodels (SARIMA) |
| Explainability | SHAP |
| Vector Search | FAISS |
| LLM Inference | Groq (Llama 3.1 8B Instant via OpenAI-compatible SDK) |

---

## Drift Monitoring

`midan/drift_monitor.py` maintains an append-only prediction log and implements a **dual-signal AND-gate** drift detector. Both signals must fire simultaneously to declare drift — a single signal is treated as noise.

- **Signal 1** — rolling mean `gap_svm` < 0.75 × baseline mean (classification margin declining)
- **Signal 2** — FCM centroid displacement ≥ 2 clusters moved > 1.5 L2 distance

A separate **macro staleness alert** fires when `STATIC_MACRO_TABLE_AS_OF` exceeds 365 days, independent of the prediction log.

---

## Design Principles

**Epistemic discipline** — every synthesis claim is bounded by the evidence quality that generated it. Observed, inferred, and speculative conclusions are explicitly distinguished in every output.

**Signal authority hierarchy** — IS is a routing signal; the router is advisory to L4; L4 is the sole decision authority. No component can bypass this chain.

**Fail-soft on missing artifacts** — all ML artifacts load as `None` on absence and return safe neutral defaults. The system degrades gracefully; it never crashes on a missing model file.

**Training source integrity** — `outer_react.py` enforces a hard contract: prediction logs never feed retraining. Only macro grid tables are valid training sources.

**Full traceability** — every L4 text field is annotated with the L1/L2/L3 signal paths that justify it via `strategic_anchors`. Every routing decision carries a `routing_basis` string.
