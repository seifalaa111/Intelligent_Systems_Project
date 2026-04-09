from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import app as midan_app

api = FastAPI(title="MIDAN AI Decision Engine API", version="1.0")

# Allow midan.html to fetch from this API
api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class IdeaRequest(BaseModel):
    idea: str
    sector: str = "Fintech"
    country: str = "EG — Egypt"

@api.post("/analyze")
async def analyze_idea(req: IdeaRequest):
    """
    Core Inference Endpoint.
    Passes the idea, sector fallback, and country fallback into the MIDAN pipeline.
    """
    if not midan_app.MODELS_LOADED:
        raise HTTPException(status_code=500, detail="MIDAN AI models failed to load.")
    
    logs = []
    
    # 1. Pipeline Routing (Agent A1)
    if req.idea and len(req.idea.strip()) > 5:
        parsed_sec, parsed_ctry, sec_found, ctry_found = midan_app.agent_a1_parse(req.idea)
        sector_key   = parsed_sec if sec_found else midan_app.SECTOR_LABEL_MAP.get(req.sector, "fintech")
        country_code = parsed_ctry if ctry_found else req.country.split(" — ")[0]
    else:
        sector_key   = midan_app.SECTOR_LABEL_MAP.get(req.sector, "fintech")
        country_code = req.country.split(" — ")[0]
        
    # 2. Run Inference Engine (Agents A2-A7)
    try:
        report = midan_app.run_inference(sector_key, country_code, logs)
        
        # 3. Format Response for Frontend
        return {
            "success": True,
            "sector": sector_key,
            "country": country_code,
            "regime": report["regime"],
            "tas_score": int(report["tas"] * 100),
            "confidence": int(report["confidence"] * 100),
            "sarima_trend": report["sarima_trend"],
            "drift_flag": report["drift_flag"],
            "action_fired": report["action_fired"],
            "report": {
                "finding": report["finding"],
                "implication": report["implication"],
                "action": report["action"],
            },
            "shap_weights": report["shap_dict"],
            "pca_coords": report["x_pca"].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:api", host="0.0.0.0", port=8000, reload=True)
