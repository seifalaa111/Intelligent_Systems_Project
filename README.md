# MIDAN — Market Intelligence & Decision Analysis Network

> An epistemically disciplined, multi-layer AI pipeline that evaluates startup ideas against live market signals, extracts structural competitive mechanisms, and delivers calibrated strategic verdicts — not just scores.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=flat-square&logo=streamlit)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## What MIDAN Does

MIDAN takes a founder's startup idea as natural language input and runs it through a structured, multi-layer reasoning pipeline. Every output field is traceable to a specific signal path. Every claim is bounded by the evidence quality that generated it.

The output is not a chatbot response. It is a structured, evidence-backed decision envelope.

---

## Architecture Overview

MIDAN is organized into four sequential reasoning layers, a mechanism extraction pipeline between L3 and L4, and an epistemic synthesis calibration layer:

```
Founder Input (natural language)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  L0 — Sanity Gate                                   │
│  3-tier rejection: unactionable / ambiguous / ok    │
│  Hard stop on ideas that cannot be evaluated        │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  L1 — Structured Feature Extraction                 │
│  LLM + heuristic fallback                           │
│  Extracts: sector, country, BM type, segment,       │
│  stage, differentiation, regulatory risk,           │
│  market readiness, competitive intensity            │
│  Each field has a confidence score                  │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  L2 — Macro Intelligence Layer                      │
│  SVM RBF → regime classification (4 classes)        │
│  LightGBM + SHAP → explainability                   │
│  SARIMA → 90-day deal volume forecast per sector    │
│  FCM (Fuzzy C-Means) → market membership scoring    │
│  DBSCAN → structural cluster assignment             │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  L3 — Structured Reasoning Modules                  │
│  differentiation · competition · business_model     │
│  unit_economics · signal_interactions               │
│  Each module: verdict + evidence trace + source tag │
│  insufficient_information state when signals thin   │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  MECHANISM EXTRACTION PIPELINE (8 phases)           │
│                                                     │
│  Phase 0   Extractability gate (mode + cap)         │
│  Phase 1A  Structural observation pass              │
│  Phase 1B  Mechanism assignment (11 types)          │
│  Phase 1C  Evidence calibration                     │
│  Phase 1D  Weight normalization                     │
│  Phase 1E  Interpretation enrichment                │
│  Phase 2   Market structure derivation              │
│  Phase 3   Tension classification (rule-based)      │
│  Phase 4   Competitive replication analysis         │
│  Phase 5   Contextual signal amplification          │
│  Phase 6   Uncertainty propagation                  │
│  Phase 7   Cross-field consistency check            │
│  Phase 8   Epistemic summary construction           │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  L4 — Decision State Machine                        │
│  Decision state: proceed / validate / revisit /     │
│                  insufficient_information           │
│  Risk decomposition: market / execution / timing    │
│  Conflict detection + severity scoring              │
│  Offsetting analysis                                │
│  mechanism_uncertainty → probabilistic L4 modifier  │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  SYNTHESIS LAYER (epistemically calibrated)         │
│  7 strategic fields: strategic_interpretation,      │
│  key_driver, main_risk, counterpoint,               │
│  differentiation_insight, what_matters_most,        │
│  counter_thesis                                     │
│  Language tone conditional on evidence_quality      │
│  epistemic_disclosure woven into prose              │
│  implication_ceiling bounds all claims              │
│  tension_coverage_state explicitly surfaced         │
└─────────────────────────────────────────────────────┘
```

---

## Mechanism Extraction

The mechanism extraction pipeline runs between L3 and L4. It extracts, calibrates, and scores up to 13 structural competitive mechanisms from the idea's signal set — then feeds the resulting uncertainty value back into L4 as a continuous probabilistic modifier.

### Mechanism Types

| Category | Mechanisms |
|---|---|
| Advantage | `network_effect` · `switching_cost` · `brand_moat` · `data_moat` · `regulatory_moat` |
| Operational | `cost_advantage` · `process_efficiency` |
| Distribution | `distribution_control` |
| Constraint | `api_dependency` · `platform_dependency` · `regulatory_headwind` |

### Epistemic Inference Ladder

Each mechanism is assigned an inference depth that determines the maximum claim level (implication ceiling) that can be made about it:

| Tier | `inference_depth` | `implication_ceiling` |
|---|---|---|
| 3 | `directly_observed` | up to `strategic_conclusion` |
| 2 | `one_step_inference` | up to `inference` |
| 1 | `speculative` | `observation` only |

Confidence calibration is quality-weighted, not count-weighted. One strong structural observation outweighs ten generic signals.

### Epistemic Summary

Every pipeline run produces an `EpistemicSummary` with:
- `evidence_quality`: `strong` / `moderate` / `weak` / `insufficient`
- `observed_signals`, `inferred_mechanisms`, `speculative_assumptions`
- `unresolved_uncertainty`, `structurally_missing`
- `recommended_disclosure`: surfaced verbatim in synthesis prose

---

## Epistemic Calibration

The synthesis layer is epistemically calibrated — language tone, claim strength, and framing are adjusted based on the mechanism pipeline's evidence quality:

| Evidence Quality | Synthesis Tone |
|---|---|
| `strong` | Decisive — produce a decision, no hedging |
| `moderate` | Calibrated — qualify inferred claims explicitly |
| `weak` | Signal-pattern framing — every claim carries epistemic qualifier |
| `insufficient` | Hypothesis-only — pattern-based, not concluded |

The `epistemic_disclosure` is woven into the `strategic_interpretation` prose directly — not buried in metadata. Claims in `differentiation_insight` are bounded by the mechanism's `implication_ceiling`. `tension_coverage_state` explicitly states the rule set evaluated and its limits.

---

## Market Regimes

| Regime | Description |
|---|---|
| `GROWTH_MARKET` | Expanding deal volume, high sector momentum |
| `EMERGING_MARKET` | Early-stage activity, increasing signal density |
| `HIGH_FRICTION_MARKET` | Active but structurally constrained |
| `CONTRACTING_MARKET` | Declining volume, adverse macro conditions |

---

## Scoring System

### L4 Decision States

| State | Meaning |
|---|---|
| `proceed` | Signals support moving forward |
| `validate` | Promising but requires specific validation |
| `revisit` | Structural issues need addressing |
| `insufficient_information` | Signal coverage too low to evaluate |

### Overall Risk

Risk decomposition across three independent dimensions: `market_risk`, `execution_risk`, `timing_risk`. Each is `low` / `moderate` / `high` with a specific driver.

`mechanism_uncertainty` (0.0 – 0.30) feeds L4 as a continuous probabilistic modifier to the overall risk level — not a hard threshold.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Frontend | Vanilla HTML/CSS/JS (`midan.html`) |
| ML Models | scikit-learn · LightGBM · statsmodels |
| Fuzzy Logic | scikit-fuzzy (FCM) |
| Explainability | SHAP |
| LLM Inference | Groq (Llama 3.1 8B Instant) |

---

## Setup

### Prerequisites
- Python 3.10+
- Groq API key (free tier at [console.groq.com](https://console.groq.com))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/seifalaa111/Intelligent_Systems_Project.git
cd Intelligent_Systems_Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate model binaries
#    Run MIDAN_Pipeline.ipynb (Jupyter or Google Colab)
#    Produces .pkl files in models/ — these are gitignored due to size

# 4. Configure environment
cp .env.example .env   # if available, or create manually
# Set: GROQ_API_KEY=your_key_here
```

---

## Running

### FastAPI backend + HTML frontend
```bash
uvicorn api:api --host 0.0.0.0 --port 8000 --reload
# Open midan.html in a browser
```

### Streamlit dashboard
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### Both simultaneously (Windows)
```powershell
.\start_servers.ps1
```

### Test suite
```bash
pytest test_pipeline_logic.py test_api.py test_interact_logic.py test_response_schema.py -v
```

---

## API Reference

### `POST /process`

Evaluate a startup idea through the full pipeline.

**Request**
```json
{
  "idea_text": "A B2B SaaS helping Egyptian logistics companies optimize last-mile delivery routing",
  "sector": "logistics",
  "country": "EG"
}
```

**Response (key fields)**
```json
{
  "decision_state": "validate",
  "overall_risk": "moderate",
  "signal_tier": "Moderate",
  "risk_decomposition": {
    "market_risk": { "level": "moderate", "driver": "..." },
    "execution_risk": { "level": "high", "driver": "..." },
    "timing_risk": { "level": "low", "driver": "..." }
  },
  "strategic_interpretation": "...",
  "key_driver": "...",
  "main_risk": "...",
  "counterpoint": "...",
  "differentiation_insight": "...",
  "epistemic_disclosure": "...",
  "mechanism_analysis": {
    "extraction_mode": "partial",
    "mechanism_count": 4,
    "mechanisms": [ ... ],
    "market_structure": { "category": "fragmented", "confidence": 0.72 },
    "tensions": [ ... ],
    "uncertainty": 0.14,
    "epistemic_summary": { ... },
    "tension_coverage_state": "..."
  }
}
```

### `POST /chat`
Conversational follow-up on a processed idea. Reads only the L4 decision envelope — never speculates beyond what the pipeline concluded.

### `GET /health`
Health check. Returns pipeline status and model availability.

---

## Project Structure

```
Intelligent_Systems_Project/
│
├── api.py                       # FastAPI app — /process, /chat, /health endpoints
├── app.py                       # Streamlit dashboard
├── midan.html                   # Static HTML/JS frontend
├── MIDAN_Pipeline.ipynb         # Training notebook — generates all model binaries
├── MIDAN_Pipeline.py            # Script version of the training pipeline
├── _split_api.py                # API splitting utility
├── requirements.txt             # Pinned Python dependencies
│
├── midan/                       # Core reasoning package
│   ├── __init__.py
│   ├── config.py                # Environment, model paths, client initialization
│   ├── core.py                  # Shared imports, type aliases, ResponsePayload schema
│   ├── pipeline.py              # Main orchestrator — runs L0→L1→L2→L3→mech→L4→synthesis
│   ├── l0_gate.py               # L0: 3-tier sanity gate with rejection codes
│   ├── l1_parser.py             # L1: feature extraction with confidence scoring
│   ├── l2_intelligence.py       # L2: SVM, SHAP, SARIMA, FCM, DBSCAN
│   ├── l3_reasoning.py          # L3: structured reasoning modules
│   ├── l4_decision.py           # L4: decision state machine + risk decomposition
│   ├── mechanism_extractor.py   # 8-phase mechanism pipeline + epistemic summary
│   ├── response.py              # Synthesis layer + epistemic calibration
│   ├── conversation.py          # Chat routing + post-decision mode branching
│   └── endpoints.py             # FastAPI route handlers
│
├── models/                      # Model artifacts
│   ├── *.pkl                    # Binary model files — gitignored (regenerate via notebook)
│   ├── sarima_results.json      # Pre-computed SARIMA forecasts
│   ├── competitors_context.json # Sector competitor context
│   ├── sentiment_context.json   # Sector sentiment context
│   └── *.png                    # Training visualization outputs
│
├── Datasets/                    # Raw training datasets
│   ├── all-data.csv
│   ├── big_startup_secsees_dataset.csv
│   ├── investments_VC.csv
│   ├── unicorn_startup_companies.csv
│   └── world_bank_data_2025.csv
│
├── test_api.py                  # API endpoint tests
├── test_pipeline_logic.py       # Pipeline unit tests
├── test_interact_logic.py       # Interaction logic tests
├── test_response_schema.py      # Response schema validation tests
├── run_tests.ps1                # PowerShell test runner
│
└── PROJECT_SCRAPPED_DATA/       # Intelligence data collection subsystem
    ├── collectors/              # Source-specific data collectors
    ├── extractors/              # Signal extraction from raw data
    ├── processors/              # Conflict resolution + schema validation
    ├── utils/                   # HTTP client, logging, text cleaning
    ├── data/                    # Collected raw, structured, training, validation data
    ├── scripts/                 # Monitoring + validation scripts
    └── main.py                  # Pipeline entry point
```

---

## Coverage

**Sectors:** Fintech · E-commerce · Healthtech · Edtech · SaaS · Logistics · Agritech

**Countries:** Egypt · Saudi Arabia · UAE · Morocco · Nigeria · Kenya · United States · United Kingdom

---

## Data Collection Subsystem

`PROJECT_SCRAPPED_DATA/` contains a standalone intelligence pipeline that collects and processes startup data from:
- Y Combinator
- Product Hunt
- Reddit
- Failory (failure post-mortems)
- Company websites

Collected data feeds the training pipeline in `MIDAN_Pipeline.ipynb`.

---

## Design Principles

- **Epistemic discipline**: every synthesis claim is bounded by the evidence quality that generated it. The system explicitly distinguishes between observed, inferred, and speculative conclusions.
- **Fail-fast by design**: L0 rejects unevaluable inputs before any ML runs. L3 reports `insufficient_information` when signal coverage is too low. The mechanism pipeline returns an uncertainty value that degrades L4 confidence probabilistically.
- **No hallucination amplification**: retrieval and synthesis cannot escalate claims beyond their mechanism's `implication_ceiling`. Synthesis tone is condition-conditional, not uniformly assertive.
- **Full traceability**: every L4 text field is paired with the L1/L2/L3 signal paths that justify it via `strategic_anchors`. The mechanism pipeline's `synthesis_trace` records every source tag.
- **Single source of truth**: `build_response_payload` in `core.py` is the only mapping function from pipeline output to API response. No field is generated outside this path.
