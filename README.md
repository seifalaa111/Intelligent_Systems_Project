# MIDAN — Market Intelligence & Decision Analysis Network

> A multi-agent AI pipeline that evaluates startup ideas against live market signals, classifies market regimes, and delivers structured verdicts before founders build.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=flat-square&logo=streamlit)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## What It Does

MIDAN takes a founder's startup idea as natural language input and runs it through a structured intelligence pipeline:

1. **Parses** the idea into structured signals (sector, country, concept)
2. **Classifies** the market regime using a trained SVM model
3. **Explains** the classification via SHAP values from a LightGBM surrogate
4. **Forecasts** 90-day deal volume trends using SARIMA per sector
5. **Scores** the idea across five evaluation dimensions
6. **Delivers** a composite verdict with a Trinity Report (Finding → Implication → Action)

The output is not a suggestion — it is a structured, evidence-backed decision signal.

---

## Pipeline Architecture

```
Founder Input (natural language)
        │
        ▼
  Agent A0 — Idea Evaluation
  (LLM-based, 5-dimension scoring)
        │
        ▼
  Agent A1 — NLP Parser
  (sector + country extraction)
        │
        ▼
  Macro Vector Construction
  (country + sector adjustment factors)
        │
        ▼
  SVM RBF Classification
  (Market Regime → 4 classes)
        │
        ├─────────────────────┐
        ▼                     ▼
  LightGBM + SHAP       SARIMA Forecast
  (explainability)      (90-day deal volume)
        │                     │
        └──────────┬──────────┘
                   ▼
         TAS Score (0–1)
         (Trend-Adjusted Score)
                   │
                   ▼
         SVS = TAS × 0.50 + Idea Score × 0.50
         (Startup Viability Score)
                   │
                   ▼
         Quadrant Verdict
         GO / Wrong Idea / Wait / STOP
                   │
                   ▼
         Trinity Report
         (Finding → Implication → Action)
                   │
                   ▼
         Slack Webhook (if TAS ≥ 0.70)
```

---

## Scoring System

### Idea Score (Agent A0)
Evaluates the startup idea across five dimensions, each scored 0–100 with reasoning:

| Dimension | Description |
|---|---|
| Problem Clarity | How well-defined and validated the problem is |
| Market Fit | Alignment between solution and target market |
| Feasibility | Technical and operational viability |
| Scalability | Potential for growth beyond initial market |
| Revenue Model | Clarity and sustainability of monetization |

### SVS — Startup Viability Score
```
SVS = TAS × 0.50 + Idea Score × 0.50
```

### Quadrant Verdict

| Verdict | Condition | Interpretation |
|---|---|---|
| **GO** | SVS ≥ 0.60 and Idea ≥ 0.50 | Strong market signal + strong idea |
| **Wrong Idea** | TAS ≥ 0.55 but Idea < 0.50 | Good market, weak concept |
| **Wait** | TAS < 0.55 but Idea ≥ 0.50 | Strong idea, market not ready |
| **STOP** | Both below threshold | Insufficient signal on both dimensions |

### Market Regimes

| Regime | Description |
|---|---|
| `GROWTH_MARKET` | Expanding deal volume, high sector momentum |
| `EMERGING_MARKET` | Early-stage activity, increasing signal density |
| `HIGH_FRICTION_MARKET` | Active but structurally constrained |
| `CONTRACTING_MARKET` | Declining volume, adverse macro conditions |

---

## Agent & Model Reference

### Agents

| Agent | Role |
|---|---|
| **A0** | Idea evaluation — LLM scoring across 5 dimensions with keyword heuristic fallback |
| **A1** | NLP parser — extracts sector and country from natural language |
| **A2** | Competitor context loader |
| **A4** | Sentiment context aggregator |
| **A6** | Trinity Report generator (Finding → Implication → Action) |
| **A7** | LLM synthesis layer (Groq / Llama3 fallback) |

### Models

| Model | Purpose |
|---|---|
| **DBSCAN** | Initial market clustering |
| **FCM** | Fuzzy membership refinement |
| **SVM RBF** | Regime classification (3 trained classes + rule-based GROWTH) |
| **LightGBM** | SHAP surrogate for decision explainability |
| **SARIMA** | Per-sector 90-day deal volume forecasting |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Frontend | Vanilla HTML/CSS/JS (`midan.html`) |
| ML Models | scikit-learn, LightGBM, statsmodels |
| Explainability | SHAP |
| LLM Inference | Groq (Llama3) |
| Fuzzy Logic | scikit-fuzzy |

---

## Setup

### Prerequisites
- Python 3.10+
- A Groq API key (free tier available at [console.groq.com](https://console.groq.com))
- Optional: Slack webhook URL for high-signal alerts

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/seifalaa111/Intelligent_Systems_Project.git
cd Intelligent_Systems_Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate model files
#    Open MIDAN_Pipeline.ipynb in Google Colab or Jupyter
#    Run all cells — produces .pkl files in models/

# 4. Configure environment variables
cp .env.example .env
# Edit .env and set:
#   GROQ_API_KEY=your_key_here
#   SLACK_WEBHOOK_URL=your_webhook_here  (optional)
```

---

## Running

### Streamlit Dashboard
Full pipeline with interactive UI:
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

### FastAPI Backend + HTML Frontend
Headless API with the `midan.html` interface:
```bash
uvicorn api:api --host 0.0.0.0 --port 8000 --reload
# Then open midan.html in a browser
```
API available at `http://localhost:8000`

### Run Tests
```bash
pytest test_pipeline_logic.py -v
```
Covers: A0 scoring, A1 parsing, regime classification rules, SVM inference, SARIMA forecasting, TAS calculation, SVS computation.

---

## Project Structure

```
Intelligent_Systems_Project/
├── api.py                      # FastAPI backend — /interact endpoint
├── app.py                      # Streamlit dashboard — full pipeline UI
├── midan.html                  # Static HTML/JS frontend
├── MIDAN_Pipeline.ipynb        # Training notebook (Colab) — generates all model files
├── test_pipeline_logic.py      # Pytest suite
├── requirements.txt            # Pinned Python dependencies
├── models/
│   ├── scaler_global.pkl       # StandardScaler (fitted on training data)
│   ├── pca_global.pkl          # PCA dimensionality reduction (5D → 2D)
│   ├── svm_global.pkl          # SVM RBF classifier
│   ├── label_encoder.pkl       # Regime label encoder
│   ├── lgb_surrogate.pkl       # LightGBM surrogate for SHAP
│   ├── sarima_results.json     # Pre-computed SARIMA forecasts per sector
│   ├── competitors_context.json
│   ├── sentiment_context.json
│   └── *.png                   # Visualization outputs from training
└── .env                        # API keys — not committed
```

---

## Coverage

**Sectors:** Fintech · E-commerce · Healthtech · Edtech · SaaS · Logistics · Agritech

**Countries:** Egypt · Saudi Arabia · UAE · Morocco · Nigeria · Kenya · United States · United Kingdom

---

## API Reference

### `POST /interact`

Evaluate a startup idea.

**Request**
```json
{
  "message": "A platform helping freelancers in Egypt receive client payments without bank transfer fees",
  "conversation_id": "optional-uuid"
}
```

**Response**
```json
{
  "response": "...",
  "market_regime": "GROWTH_MARKET",
  "tas_score": 0.74,
  "idea_score": 0.68,
  "svs_score": 0.71,
  "verdict": "GO",
  "shap_explanation": "...",
  "trinity_report": {
    "finding": "...",
    "implication": "...",
    "action": "..."
  }
}
```
