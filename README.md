# MIDAN — AI Decision Engine

Multi-agent AI pipeline for startup market intelligence. Classifies market regimes, explains decisions, forecasts sector trends, and computes an actionable opportunity score.

## Pipeline Architecture

```
Founder Input → Agent A0 (Idea Evaluation)
                    ↓
              Agent A1 (NLP Parser)
                    ↓
              Macro Vector Construction (country + sector adjustments)
                    ↓
              SVM RBF Classification → Market Regime
                    ↓
         ┌─────────┴─────────┐
    SHAP (LightGBM)     SARIMA Forecast
         └─────────┬─────────┘
                    ↓
              TAS Score (Market Opportunity)
                    ↓
              SVS = TAS × 0.50 + Idea Score × 0.50
                    ↓
         Quadrant Verdict (GO / Wrong Idea / Wait / STOP)
                    ↓
         Trinity Report (Finding → Implication → Action)
                    ↓
              Slack Webhook (if TAS ≥ 0.70)
```

**Agents:**
- **A0** — Idea Evaluation (LLM-based + keyword heuristic fallback, scores 5 dimensions)
- **A1** — Keyword-based NLP parser (sector + country extraction)
- **A2** — Competitor context loader
- **A4** — Sentiment context aggregator
- **A6** — Trinity Report generator
- **A7** — LLM synthesis (Groq / Llama3 fallback)

**Models:**
- **DBSCAN** → initial clustering
- **FCM** → fuzzy membership refinement
- **SVM RBF** → regime classification (3 classes + rule-based GROWTH)
- **LightGBM** → surrogate for SHAP explainability
- **SARIMA** → per-sector 90-day deal volume forecast

**Market Regimes:** GROWTH_MARKET, EMERGING_MARKET, HIGH_FRICTION_MARKET, CONTRACTING_MARKET

**Idea Evaluation (Agent A0):**
- Scores startup ideas across 5 dimensions: Problem Clarity, Market Fit, Feasibility, Scalability, Revenue Model
- Each dimension scored 0–100 with reasoning
- Combined into an Idea Score (0–1)
- **SVS (Startup Viability Score)** = TAS × 0.50 + Idea Score × 0.50
- **Quadrant Verdict:**
  - **GO** — SVS ≥ 0.60 and Idea ≥ 0.50 (strong market + strong idea)
  - **Wrong Idea** — TAS ≥ 0.55 but Idea < 0.50 (good market, weak idea)
  - **Wait** — TAS < 0.55 but Idea ≥ 0.50 (weak market, strong idea)
  - **STOP** — both below thresholds

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate model files (run the notebook first)
#    Open MIDAN_Pipeline.ipynb in Google Colab or Jupyter
#    and run all cells — this creates the .pkl files in models/

# 3. (Optional) Set environment variables
#    Create a .env file with your GROQ_API_KEY and SLACK_WEBHOOK_URL
```

## Running

### Streamlit Dashboard (main UI)
```bash
streamlit run app.py
```

### FastAPI Backend (for midan.html frontend)
```bash
uvicorn api:api --host 0.0.0.0 --port 8000 --reload
# Then open midan.html in a browser
```

### Run Tests
```bash
pytest test_pipeline_logic.py -v
```

## Project Structure

```
├── app.py                    # Streamlit dashboard (full pipeline + UI)
├── api.py                    # FastAPI endpoint for midan.html
├── midan.html                # Static HTML frontend
├── test_pipeline_logic.py    # Pytest suite (A0, A1, regime rules, SVM, SARIMA, TAS, SVS)
├── MIDAN_Pipeline.ipynb      # Training notebook (Colab) — generates all models
├── requirements.txt          # Pinned Python dependencies
├── models/
│   ├── scaler_global.pkl     # StandardScaler
│   ├── pca_global.pkl        # PCA (5D → 2D)
│   ├── svm_global.pkl        # SVM RBF classifier
│   ├── label_encoder.pkl     # Regime label encoder
│   ├── lgb_surrogate.pkl     # LightGBM (SHAP surrogate)
│   ├── sarima_results.json   # Per-sector SARIMA forecasts
│   ├── competitors_context.json
│   ├── sentiment_context.json
│   └── *.png                 # Visualization outputs
└── .env                      # API keys (not committed)
```

## Supported Sectors & Countries

**Sectors:** Fintech, E-commerce, Healthtech, Edtech, SaaS, Logistics, Agritech

**Countries:** Egypt (EG), Saudi Arabia (SA), UAE (AE), Morocco (MA), Nigeria (NG), Kenya (KE), United States (US), United Kingdom (GB)
