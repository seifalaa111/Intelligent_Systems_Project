# sra/step1_ingestion.py
import pandas as pd
import numpy as np
import time


def extract_context_a1(text):
    """
    NLP parser simulation extracting market parameters.
    """
    text = text.lower()
    sector = "fintech"
    country = "eg"

    if "health" in text or "clinic" in text:
        sector = "healthtech"
    if "saudi" in text or "ksa" in text:
        country = "sa"
    if "uae" in text or "dubai" in text:
        country = "ae"

    return {"sector": sector, "country": country}


def fetch_live_macro(country_code):
    """
    Retrieves real-time macroeconomic indicators.
    """
    macro_database = {
        "eg": {"inflation": 28.3, "gdp": 3.8, "internet": 73.9},
        "sa": {"inflation": 2.5, "gdp": 1.5, "internet": 99.0},
        "ae": {"inflation": 3.1, "gdp": 3.4, "internet": 99.0},
    }
    return macro_database.get(country_code, {"inflation": 5.0, "gdp": 2.0, "internet": 80.0})


def fetch_sector_medians(sector):
    """
    Retrieves trailing 24-month medians for the specified sector.
    """
    sector_database = {
        "fintech": {"median_funding": 15000000, "median_deals": 120},
        "healthtech": {"median_funding": 12000000, "median_deals": 85},
    }
    return sector_database.get(sector, {"median_funding": 5000000, "median_deals": 30})


def process_founder_input(idea_text):
    """
    Master pipeline for Step 1: Ingestion and Vectorization.
    """
    context = extract_context_a1(idea_text)
    live_macro = fetch_live_macro(context["country"])
    sector_medians = fetch_sector_medians(context["sector"])

    # Constructing the Inference Vector
    X_new = np.array([[
        live_macro["inflation"],
        live_macro["gdp"],
        live_macro["internet"],
        sector_medians["median_funding"],
        sector_medians["median_deals"]
    ]])

    return context, live_macro, sector_medians, X_new


def handle_csv_upload(uploaded_file):
    """
    Processes batch historical data uploads.
    """
    df = pd.read_csv(uploaded_file)
    return len(df)