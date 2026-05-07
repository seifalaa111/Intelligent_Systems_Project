from fastapi.testclient import TestClient

import api


client = TestClient(api.api)


def test_health_endpoint_reports_service_status():
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data


def test_analyze_endpoint_returns_structured_analysis():
    response = client.post(
        "/analyze",
        json={
            "idea": "Invoice financing for Egyptian SMEs with faster underwriting.",
            "sector": "Fintech",
            "country": "EG â€” Egypt",
        },
    )
    assert response.status_code == 200

    data = response.json()
    # New canonical contract: ResponsePayload schema
    assert data["success"] is True
    assert data["schema_version"] == "1.0"
    assert data["decision_state"] in ("GO", "CONDITIONAL", "NO_GO")
    assert data["decision_strength"]["tier"] in ("strong", "moderate", "weak", "uncertain")
    assert "market_risk"    in data["risk_decomposition"]
    assert "execution_risk" in data["risk_decomposition"]
    assert "timing_risk"    in data["risk_decomposition"]
    # Legacy raw pipeline output is preserved under raw_pipeline_output / data
    raw = data["raw_pipeline_output"]
    assert raw["sector"] == "fintech"
    assert raw["country"] == "EG"
    assert "tas_score" in raw
    assert "idea_features" in raw


def test_analyze_endpoint_rejects_too_short_idea():
    response = client.post(
        "/analyze",
        json={"idea": "hi", "sector": "Fintech", "country": "EG â€” Egypt"},
    )
    assert response.status_code == 400


def test_project_endpoint_refuses_generic_input_without_faking_structure():
    response = client.post("/project", json={"idea": "fintech"})
    assert response.status_code == 200

    data = response.json()
    # `type` replaces legacy `mode`; PRE_ANALYSIS replaces analysis_allowed=False
    assert data["type"] == "interpretation"
    assert data["decision_state"] == "PRE_ANALYSIS"
    assert data["projection"]["quality"]["components"]["mechanism"] is False
    assert "What is the product exactly?" in data["projection"]["guidance"]


def test_project_endpoint_builds_analysis_for_structured_input():
    response = client.post(
        "/project",
        json={
            "idea": "We help independent restaurants in Cairo cut food waste with AI demand forecasting and supplier planning."
        },
    )
    assert response.status_code == 200

    data = response.json()
    assert data["type"] == "analysis"
    assert data["decision_state"] in ("GO", "CONDITIONAL", "NO_GO")
    assert data["projection"]["signal_summary"]["sector"] == "saas"
    assert data["projection"]["signal_summary"]["dominant_risk"] == "workflow"


def test_project_probe_answers_directly():
    response = client.post(
        "/project",
        json={
            "idea": "We help independent restaurants in Cairo cut food waste with AI demand forecasting and supplier planning.",
            "question": "why is the idea bad",
            "context": {
                "idea": "We help independent restaurants in Cairo cut food waste with AI demand forecasting and supplier planning.",
                "main_risk": "Restaurants may not change a live planning workflow without proof of immediate ROI.",
                "counter_thesis": "Forecasting is not enough if the buyer still trusts instinct more than software."
            },
        },
    )
    assert response.status_code == 200

    data = response.json()
    # Probe answer is now the canonical `reply`; `type` replaces legacy `mode`.
    assert data["type"] == "probe"
    assert data["reply"].lower().startswith("the idea is weak because")
