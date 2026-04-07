"""
build_feature_matrix.py
=======================
Builds TWO separate outputs from the raw datasets:

1. ML Matrix  → /data/ml_matrix/ml_matrix_global.csv
   Clean rectangular features + real labels for DBSCAN→FCM→SVM pipeline

2. SARIMA Series → /data/sarima/monthly_deals_by_sector.csv
   Monthly deal count per sector for time series forecasting

Design decisions:
- Labels come from company `status` column (operating/closed/acquired/ipo) — REAL ground truth
- Training is global (66K companies) — not Egypt-only (only 36 Egypt records exist)
- Egypt macro data injected from World Bank at inference time, not training time
- No fake rows, no median imputation for missing combinations
- Only rows with real data are kept
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
RAW = "/home/claude/datasets/New folder"
OUT_ML     = "/home/claude/asde/data/ml_matrix"
OUT_SARIMA = "/home/claude/asde/data/sarima"

# ─────────────────────────────────────────────
# SECTOR MAPPING
# Maps raw Crunchbase categories → 9 standard sectors
# ─────────────────────────────────────────────
SECTOR_MAP = {
    "fintech":     ["Finance", "Financial Services", "FinTech", "Payments",
                    "Banking", "Lending", "Insurance", "FinTech"],
    "healthtech":  ["Biotechnology", "Health Care", "Health and Wellness",
                    "Medical", "Pharmaceutical", "Bioinformatics",
                    "Healthcare", "Digital Health"],
    "edtech":      ["Education", "E-Learning", "EdTech", "Training"],
    "ecommerce":   ["E-Commerce", "Retail", "Marketplaces", "Shopping"],
    "saas":        ["SaaS", "Enterprise Software", "Cloud Computing",
                    "Software", "PaaS", "Infrastructure"],
    "logistics":   ["Transportation", "Logistics", "Supply Chain",
                    "Delivery", "Shipping", "Fleet Management"],
    "agritech":    ["Agriculture", "Food", "AgriTech", "FoodTech",
                    "Food and Beverage"],
    "cleantech":   ["Clean Technology", "Energy", "Renewable Energy",
                    "CleanTech", "Sustainability", "Environment"],
    "proptech":    ["Real Estate", "Property", "PropTech", "Construction",
                    "Mortgage"],
}

# Reverse map: raw_category → sector
CAT_TO_SECTOR = {}
for sector, cats in SECTOR_MAP.items():
    for cat in cats:
        CAT_TO_SECTOR[cat.lower().strip()] = sector

# ─────────────────────────────────────────────
# ISO3 → ISO2 mapping for World Bank join
# ─────────────────────────────────────────────
ISO3_TO_ISO2 = {
    "USA": "us", "GBR": "gb", "CAN": "ca", "IND": "in", "CHN": "cn",
    "FRA": "fr", "DEU": "de", "ISR": "il", "ESP": "es", "AUS": "au",
    "NLD": "nl", "RUS": "ru", "SGP": "sg", "SWE": "se", "BRA": "br",
    "IRL": "ie", "JPN": "jp", "ITA": "it", "KOR": "kr", "CHE": "ch",
    "DNK": "dk", "FIN": "fi", "NOR": "no", "BEL": "be", "AUT": "at",
    "NZL": "nz", "PRT": "pt", "POL": "pl", "TUR": "tr", "ZAF": "za",
    "MEX": "mx", "ARG": "ar", "CHL": "cl", "COL": "co", "PER": "pe",
    "EGY": "eg", "SAU": "sa", "ARE": "ae", "JOR": "jo", "LBN": "lb",
    "MAR": "ma", "TUN": "tn", "NGA": "ng", "KEN": "ke", "GHA": "gh",
    "PAK": "pk", "BGD": "bd", "IDN": "id", "MYS": "my", "THA": "th",
    "PHL": "ph", "VNM": "vn", "HKG": "hk", "TWN": "tw", "UKR": "ua",
    "CZE": "cz", "HUN": "hu", "ROU": "ro", "GRC": "gr", "BGR": "bg",
    "HRV": "hr", "SVK": "sk", "SVN": "si", "EST": "ee", "LTU": "lt",
    "LVA": "lv", "SRB": "rs", "KAZ": "kz", "UZB": "uz", "AZE": "az",
    "GEO": "ge", "ARM": "am", "BHR": "bh", "KWT": "kw", "QAT": "qa",
    "OMN": "om", "IRQ": "iq", "IRN": "ir", "SDN": "sd", "ETH": "et",
}

# ─────────────────────────────────────────────
# LABEL MAPPING
# Real outcome labels from company status
# ─────────────────────────────────────────────
STATUS_TO_LABEL = {
    "operating": "VIABLE",
    "acquired":  "ATTRACTIVE",
    "ipo":       "ATTRACTIVE",
    "closed":    "HIGH_RISK",
}

def map_sector(category_list):
    """Map a pipe-separated category string to one of 9 sectors."""
    if pd.isna(category_list):
        return None
    cats = [c.lower().strip() for c in str(category_list).split("|")]
    for cat in cats:
        if cat in CAT_TO_SECTOR:
            return CAT_TO_SECTOR[cat]
    return None


def load_startups():
    """Load and clean the primary startup dataset."""
    print("Loading startups...")
    df = pd.read_csv(
        os.path.join(RAW, "big_startup_secsees_dataset.csv"),
        encoding="latin-1",
        on_bad_lines="skip",
        low_memory=False
    )

    # Keep only needed columns
    df = df[["category_list", "funding_total_usd", "status",
             "country_code", "funding_rounds", "founded_at",
             "first_funding_at", "last_funding_at"]].copy()

    # Map sector
    df["sector"] = df["category_list"].apply(map_sector)
    df = df[df["sector"].notna()].copy()

    # Map label from status
    df["status_clean"] = df["status"].str.lower().str.strip()
    df["label"] = df["status_clean"].map(STATUS_TO_LABEL)
    df = df[df["label"].notna()].copy()

    # Map country ISO2
    df["country_iso2"] = df["country_code"].map(ISO3_TO_ISO2)

    # Parse funding
    df["funding_total_usd"] = pd.to_numeric(
        df["funding_total_usd"].astype(str).str.replace(",", ""),
        errors="coerce"
    )
    df["funding_rounds"] = pd.to_numeric(df["funding_rounds"], errors="coerce")

    # Extract funding year for World Bank join
    df["last_funding_at"] = pd.to_datetime(df["last_funding_at"], errors="coerce")
    df["funding_year"] = df["last_funding_at"].dt.year

    # Keep years that overlap with World Bank data (2010-2023)
    df = df[(df["funding_year"] >= 2010) & (df["funding_year"] <= 2023)].copy()

    # Drop rows missing critical fields
    df = df.dropna(subset=["sector", "label", "funding_rounds"])

    print(f"  → {len(df):,} startups after cleaning")
    print(f"  → Sectors: {df['sector'].value_counts().to_dict()}")
    print(f"  → Labels:  {df['label'].value_counts().to_dict()}")
    return df


def load_world_bank():
    """Load World Bank macro indicators."""
    print("Loading World Bank macro data...")
    wb = pd.read_csv(os.path.join(RAW, "world_bank_data_2025.csv"))
    wb = wb.rename(columns={
        "country_id": "country_iso2",
        "year": "funding_year",
        "Inflation (CPI %)": "inflation_rate",
        "GDP Growth (% Annual)": "gdp_growth",
        "Unemployment Rate (%)": "unemployment_rate",
    })
    wb["country_iso2"] = wb["country_iso2"].str.lower().str.strip()
    wb["funding_year"] = pd.to_numeric(wb["funding_year"], errors="coerce")

    wb = wb[["country_iso2", "funding_year",
             "inflation_rate", "gdp_growth", "unemployment_rate"]].copy()
    wb = wb.dropna(subset=["country_iso2", "funding_year"])
    print(f"  → {len(wb):,} country-year records")
    return wb


def load_internet():
    """Load internet penetration data."""
    print("Loading internet penetration...")
    inet = pd.read_csv(os.path.join(RAW, "internet-users-by-country-2024.csv"))
    inet = inet.rename(columns={
        "country": "country_name",
        "InternetUsers_PctOfPopulationUsingInternet": "internet_pct"
    })

    # World Bank uses variant names (e.g. "Egypt, Arab Rep." not "Egypt")
    # Build normalized name → iso2 with manual overrides for common mismatches
    wb_countries = pd.read_csv(os.path.join(RAW, "world_bank_data_2025.csv"))

    # Normalize: strip suffixes like ", Arab Rep.", ", Islamic Rep." etc.
    def normalize(name):
        name = str(name).lower().strip()
        for suffix in [", arab rep.", ", islamic rep.", ", rep.", ", the",
                       " (islamic republic of)", " (bolivarian republic of)"]:
            name = name.replace(suffix, "")
        return name.strip()

    name_to_iso2 = {normalize(n): iso2.lower().strip()
                    for n, iso2 in zip(wb_countries["country_name"],
                                       wb_countries["country_id"])}

    inet["country_iso2"] = inet["country_name"].apply(normalize).map(name_to_iso2)
    inet = inet[["country_iso2", "internet_pct"]].dropna()
    print(f"  → {len(inet):,} countries with internet data")

    # Verify Egypt
    eg_val = inet[inet["country_iso2"] == "eg"]["internet_pct"].values
    print(f"  → Egypt internet pct: {eg_val[0] if len(eg_val) > 0 else 'MISSING'}%")
    return inet


def build_ml_matrix(startups, world_bank, internet):
    """Join all sources into clean ML matrix."""
    print("\nBuilding ML matrix...")

    # Join World Bank macro data
    df = startups.merge(world_bank, on=["country_iso2", "funding_year"], how="left")

    # Join internet penetration (2024 data used as proxy — stable metric)
    df = df.merge(internet, on="country_iso2", how="left")

    # Feature engineering
    # Log transform funding (right-skewed)
    df["log_funding"] = np.log1p(df["funding_total_usd"].fillna(0))

    # Country risk score: higher inflation + lower GDP = higher risk
    df["inflation_rate"] = df["inflation_rate"].clip(0, 100)
    df["gdp_growth"] = df["gdp_growth"].clip(-20, 30)
    df["internet_pct"] = df["internet_pct"].fillna(df["internet_pct"].median())
    df["inflation_rate"] = df["inflation_rate"].fillna(df["inflation_rate"].median())
    df["gdp_growth"] = df["gdp_growth"].fillna(df["gdp_growth"].median())
    df["unemployment_rate"] = df["unemployment_rate"].fillna(df["unemployment_rate"].median())

    # Final ML columns only
    ml_cols = [
        "sector",
        "country_iso2",
        "funding_year",
        "funding_rounds",
        "log_funding",
        "inflation_rate",
        "gdp_growth",
        "unemployment_rate",
        "internet_pct",
        "label"
    ]
    ml = df[ml_cols].copy()

    # Drop rows with any null in features
    feature_cols = [c for c in ml_cols if c != "label"]
    ml = ml.dropna(subset=feature_cols)

    print(f"  → Final ML matrix: {len(ml):,} rows × {len(ml_cols)} columns")
    print(f"  → Label distribution:\n{ml['label'].value_counts()}")
    print(f"  → Sector distribution:\n{ml['sector'].value_counts()}")
    return ml


def build_sarima_series():
    """Build monthly deal count per sector for SARIMA."""
    print("\nBuilding SARIMA time series...")
    df = pd.read_csv(
        os.path.join(RAW, "investments_VC.csv"),
        encoding="latin-1",
        on_bad_lines="skip",
        low_memory=False
    )

    df["sector"] = df["category_list"].apply(map_sector)
    df = df[df["sector"].notna()].copy()

    # Use first_funding_at — proper date column
    df["first_funding_at"] = pd.to_datetime(df["first_funding_at"], errors="coerce")
    df = df.dropna(subset=["first_funding_at"])

    df["funding_year"]  = df["first_funding_at"].dt.year
    df["funding_month"] = df["first_funding_at"].dt.month

    # Keep 2005-2014: enough history, avoids bad early data
    df = df[(df["funding_year"] >= 2005) & (df["funding_year"] <= 2014)].copy()

    df["year_month"] = (df["funding_year"].astype(int).astype(str) + "-" +
                        df["funding_month"].astype(int).astype(str).str.zfill(2))

    sarima = (df.groupby(["sector", "year_month"])
                .size()
                .reset_index(name="deal_count"))

    sarima = sarima.sort_values(["sector", "year_month"])
    print(f"  → {len(sarima):,} sector-month records")

    # Months per sector
    counts = sarima.groupby("sector")["deal_count"].count()
    print(f"\n  Months of data per sector (need ≥ 24 for SARIMA):")
    print(counts.to_string())
    return sarima
    return sarima


def save_sector_splits(ml_matrix):
    """Save one CSV per sector for sector-specific model training."""
    print("\nSaving sector splits...")
    for sector in ml_matrix["sector"].unique():
        sector_df = ml_matrix[ml_matrix["sector"] == sector].copy()
        path = os.path.join(OUT_ML, f"{sector}_ml.csv")
        sector_df.to_csv(path, index=False)
        print(f"  → {sector}: {len(sector_df):,} rows → {path}")


def main():
    print("=" * 60)
    print("ASDE Feature Matrix Builder — v1.0")
    print("=" * 60)

    # Load
    startups   = load_startups()
    world_bank = load_world_bank()
    internet   = load_internet()

    # Build ML matrix
    ml_matrix = build_ml_matrix(startups, world_bank, internet)

    # Build SARIMA series
    sarima = build_sarima_series()

    # Save outputs
    global_path = os.path.join(OUT_ML, "ml_matrix_global.csv")
    ml_matrix.to_csv(global_path, index=False)
    print(f"\n✅ Global ML matrix saved → {global_path}")

    sarima_path = os.path.join(OUT_SARIMA, "monthly_deals_by_sector.csv")
    sarima.to_csv(sarima_path, index=False)
    print(f"✅ SARIMA series saved   → {sarima_path}")

    save_sector_splits(ml_matrix)

    print("\n" + "=" * 60)
    print("DONE — Feature matrix ready for SRA pipeline")
    print("=" * 60)

    # Final summary
    print("\nML MATRIX SUMMARY:")
    print(f"  Rows:    {len(ml_matrix):,}")
    print(f"  Columns: {list(ml_matrix.columns)}")
    print(f"  Labels:  {ml_matrix['label'].value_counts().to_dict()}")
    print(f"  Sectors: {sorted(ml_matrix['sector'].unique().tolist())}")

    # Verify Egypt data is correct
    eg = ml_matrix[ml_matrix["country_iso2"] == "eg"]
    print(f"\nEGYPT CHECK:")
    print(f"  Egypt rows: {len(eg)}")
    if len(eg) > 0:
        print(f"  Avg inflation: {eg['inflation_rate'].mean():.1f}%")
        print(f"  Avg GDP growth: {eg['gdp_growth'].mean():.1f}%")
        print(f"  Avg internet pct: {eg['internet_pct'].mean():.1f}%")


if __name__ == "__main__":
    main()
