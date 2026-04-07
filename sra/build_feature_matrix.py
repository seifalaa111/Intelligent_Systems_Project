import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

RAW        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
OUT_ML     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ml_matrix")
OUT_SARIMA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sarima")

os.makedirs(OUT_ML,     exist_ok=True)
os.makedirs(OUT_SARIMA, exist_ok=True)

SECTOR_MAP = {
    "fintech":     ["Finance", "Financial Services", "FinTech", "Payments", "Banking", "Lending", "Insurance"],
    "healthtech":  ["Biotechnology", "Health Care", "Health and Wellness", "Medical", "Pharmaceutical", "Bioinformatics", "Healthcare", "Digital Health"],
    "edtech":      ["Education", "E-Learning", "EdTech", "Training"],
    "ecommerce":   ["E-Commerce", "Retail", "Marketplaces", "Shopping"],
    "saas":        ["SaaS", "Enterprise Software", "Cloud Computing", "Software", "PaaS", "Infrastructure"],
    "logistics":   ["Transportation", "Logistics", "Supply Chain", "Delivery", "Shipping", "Fleet Management"],
    "agritech":    ["Agriculture", "Food", "AgriTech", "FoodTech", "Food and Beverage"],
    "cleantech":   ["Clean Technology", "Energy", "Renewable Energy", "CleanTech", "Sustainability", "Environment"],
    "proptech":    ["Real Estate", "Property", "PropTech", "Construction", "Mortgage"],
}

CAT_TO_SECTOR = {}
for sector, cats in SECTOR_MAP.items():
    for cat in cats:
        CAT_TO_SECTOR[cat.lower().strip()] = sector

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

STATUS_TO_LABEL = {
    "operating": "VIABLE",
    "acquired":  "ATTRACTIVE",
    "ipo":       "ATTRACTIVE",
    "closed":    "HIGH_RISK",
}

def map_sector(category_list):
    if pd.isna(category_list):
        return None
    cats = [c.lower().strip() for c in str(category_list).split("|")]
    for cat in cats:
        if cat in CAT_TO_SECTOR:
            return CAT_TO_SECTOR[cat]
    return None

def load_startups():
    print("Loading startups...")
    df = pd.read_csv(os.path.join(RAW, "big_startup_secsees_dataset.csv"), encoding="latin-1", on_bad_lines="skip", low_memory=False)
    df = df[["category_list", "funding_total_usd", "status", "country_code", "funding_rounds", "founded_at", "first_funding_at", "last_funding_at"]].copy()
    df["sector"] = df["category_list"].apply(map_sector)
    df = df[df["sector"].notna()].copy()
    df["status_clean"] = df["status"].str.lower().str.strip()
    df["label"] = df["status_clean"].map(STATUS_TO_LABEL)
    df = df[df["label"].notna()].copy()
    df["country_iso2"] = df["country_code"].map(ISO3_TO_ISO2)
    df["funding_total_usd"] = pd.to_numeric(df["funding_total_usd"].astype(str).str.replace(",", ""), errors="coerce")
    df["funding_rounds"] = pd.to_numeric(df["funding_rounds"], errors="coerce")
    df["last_funding_at"] = pd.to_datetime(df["last_funding_at"], errors="coerce")
    df["funding_year"] = df["last_funding_at"].dt.year
    df = df[(df["funding_year"] >= 2010) & (df["funding_year"] <= 2023)].copy()
    df = df.dropna(subset=["sector", "label", "funding_rounds"])
    print(f"  -> {len(df):,} startups after cleaning")
    print(f"  -> Labels: {df['label'].value_counts().to_dict()}")
    return df

def load_world_bank():
    print("Loading World Bank macro data...")
    wb = pd.read_csv(os.path.join(RAW, "world_bank_data_2025.csv"))
    wb = wb.rename(columns={"country_id": "country_iso2", "year": "funding_year", "Inflation (CPI %)": "inflation_rate", "GDP Growth (% Annual)": "gdp_growth", "Unemployment Rate (%)": "unemployment_rate"})
    wb["country_iso2"] = wb["country_iso2"].str.lower().str.strip()
    wb["funding_year"] = pd.to_numeric(wb["funding_year"], errors="coerce")
    wb = wb[["country_iso2", "funding_year", "inflation_rate", "gdp_growth", "unemployment_rate"]].copy()
    wb = wb.dropna(subset=["country_iso2", "funding_year"])
    print(f"  -> {len(wb):,} country-year records")
    return wb

def load_internet():
    print("Loading internet penetration...")
    inet = pd.read_csv(os.path.join(RAW, "internet-users-by-country-2024.csv"))
    inet = inet.rename(columns={"country": "country_name", "InternetUsers_PctOfPopulationUsingInternet": "internet_pct"})
    wb_countries = pd.read_csv(os.path.join(RAW, "world_bank_data_2025.csv"))
    def normalize(name):
        name = str(name).lower().strip()
        for suffix in [", arab rep.", ", islamic rep.", ", rep.", ", the", " (islamic republic of)", " (bolivarian republic of)"]:
            name = name.replace(suffix, "")
        return name.strip()
    name_to_iso2 = {normalize(n): iso2.lower().strip() for n, iso2 in zip(wb_countries["country_name"], wb_countries["country_id"])}
    inet["country_iso2"] = inet["country_name"].apply(normalize).map(name_to_iso2)
    inet = inet[["country_iso2", "internet_pct"]].dropna()
    eg_val = inet[inet["country_iso2"] == "eg"]["internet_pct"].values
    print(f"  -> Egypt internet: {eg_val[0] if len(eg_val) > 0 else 'MISSING'}%")
    return inet

def build_ml_matrix(startups, world_bank, internet):
    print("\nBuilding ML matrix...")
    df = startups.merge(world_bank, on=["country_iso2", "funding_year"], how="left")
    df = df.merge(internet, on="country_iso2", how="left")
    df["log_funding"] = np.log1p(df["funding_total_usd"].fillna(0))
    df["inflation_rate"]    = df["inflation_rate"].clip(0, 100).fillna(df["inflation_rate"].median())
    df["gdp_growth"]        = df["gdp_growth"].clip(-20, 30).fillna(df["gdp_growth"].median())
    df["internet_pct"]      = df["internet_pct"].fillna(df["internet_pct"].median())
    df["unemployment_rate"] = df["unemployment_rate"].fillna(df["unemployment_rate"].median())
    ml_cols = ["sector", "country_iso2", "funding_year", "funding_rounds", "log_funding", "inflation_rate", "gdp_growth", "unemployment_rate", "internet_pct", "label"]
    ml = df[ml_cols].copy()
    ml = ml.dropna(subset=[c for c in ml_cols if c != "label"])
    print(f"  -> ML matrix: {len(ml):,} rows x {len(ml_cols)} columns")
    print(f"  -> Labels:\n{ml['label'].value_counts()}")
    return ml

def build_sarima_series():
    print("\nBuilding SARIMA time series...")
    df = pd.read_csv(os.path.join(RAW, "investments_VC.csv"), encoding="latin-1", on_bad_lines="skip", low_memory=False)
    df["sector"] = df["category_list"].apply(map_sector)
    df = df[df["sector"].notna()].copy()
    df["first_funding_at"] = pd.to_datetime(df["first_funding_at"], errors="coerce")
    df = df.dropna(subset=["first_funding_at"])
    df["funding_year"]  = df["first_funding_at"].dt.year
    df["funding_month"] = df["first_funding_at"].dt.month
    df = df[(df["funding_year"] >= 2005) & (df["funding_year"] <= 2014)].copy()
    df["year_month"] = df["funding_year"].astype(int).astype(str) + "-" + df["funding_month"].astype(int).astype(str).str.zfill(2)
    sarima = df.groupby(["sector", "year_month"]).size().reset_index(name="deal_count").sort_values(["sector", "year_month"])
    print(f"  -> {len(sarima):,} sector-month records")
    counts = sarima.groupby("sector")["deal_count"].count()
    print(f"  -> Months per sector:\n{counts.to_string()}")
    return sarima

def save_sector_splits(ml_matrix):
    print("\nSaving sector splits...")
    for sector in ml_matrix["sector"].unique():
        path = os.path.join(OUT_ML, f"{sector}_ml.csv")
        ml_matrix[ml_matrix["sector"] == sector].to_csv(path, index=False)
        print(f"  -> {sector}: {len(ml_matrix[ml_matrix['sector'] == sector]):,} rows")

def main():
    print("=" * 60)
    print("ASDE Feature Matrix Builder v1.0")
    print("=" * 60)
    startups   = load_startups()
    world_bank = load_world_bank()
    internet   = load_internet()
    ml_matrix  = build_ml_matrix(startups, world_bank, internet)
    sarima     = build_sarima_series()
    ml_matrix.to_csv(os.path.join(OUT_ML, "ml_matrix_global.csv"), index=False)
    sarima.to_csv(os.path.join(OUT_SARIMA, "monthly_deals_by_sector.csv"), index=False)
    save_sector_splits(ml_matrix)
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    eg = ml_matrix[ml_matrix["country_iso2"] == "eg"]
    print(f"\nEGYPT CHECK: {len(eg)} rows")
    if len(eg) > 0:
        print(f"  inflation:  {eg['inflation_rate'].mean():.1f}%")
        print(f"  gdp_growth: {eg['gdp_growth'].mean():.1f}%")
        print(f"  internet:   {eg['internet_pct'].mean():.1f}%")

if __name__ == "__main__":
    main()
