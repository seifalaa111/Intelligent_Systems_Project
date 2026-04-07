import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sector_mapping import SECTORS, COUNTRIES, map_sector_by_keywords, standardize_columns
import warnings
warnings.filterwarnings('ignore')

def build_spine():
    months = pd.date_range('2010-01', '2024-12', freq='MS')
    rows = [{'sector': s, 'country': c, 'year_month': m.strftime('%Y-%m')} for s in SECTORS for c in COUNTRIES for m in months]
    return pd.DataFrame(rows)

def load_funding():
    # FIXED ENCODING HERE
    df = standardize_columns(pd.read_csv('data/raw/funding_2024.csv', encoding='latin-1', encoding_errors='replace'))
    df['sector'] = df['category_list'].apply(map_sector_by_keywords)
    df['country'] = df['country_code']
    df['year_month'] = pd.to_datetime(df['founded_at'], errors='coerce').dt.strftime('%Y-%m')
    df = df.dropna(subset=['sector', 'country', 'year_month', 'funding_total_usd'])
    df['funding_amount_usd'] = pd.to_numeric(df['funding_total_usd'], errors='coerce').fillna(0)
    
    res = df.groupby(['sector', 'country', 'year_month']).agg(
        funding_amount_usd=('funding_amount_usd', 'sum'),
        deal_count_24m=('permalink', 'count')
    ).reset_index()
    
    res['market_cagr_pct'] = res.groupby(['sector', 'country'])['funding_amount_usd'].pct_change(24).fillna(0) * 100
    return res

def load_outcomes():
    df = standardize_columns(pd.read_csv('data/raw/crunchbase_outcomes.csv'))
    df['sector'] = df['category_list'].apply(map_sector_by_keywords)
    df['country'] = df['country_code']
    df['year_month'] = pd.to_datetime(df['founded_at'], errors='coerce').dt.strftime('%Y-%m')
    df = df.dropna(subset=['sector', 'country', 'year_month'])
    
    res = df.groupby(['sector', 'country', 'year_month']).agg(
        competitor_count=('permalink', 'count')
    ).reset_index()
    return res

def load_jobs():
    # Only load necessary columns to save RAM from the 500MB file
    df = standardize_columns(pd.read_csv('data/raw/linkedin_jobs.csv', usecols=['title', 'location', 'original_listed_time']))
    df['sector'] = df['title'].apply(map_sector_by_keywords)
    df['country'] = df['location'].str[-2:].str.upper() # Rough country extraction
    df['year_month'] = pd.to_datetime(df['original_listed_time'], unit='ms', errors='coerce').dt.strftime('%Y-%m')
    df = df.dropna(subset=['sector', 'year_month'])
    
    res = df.groupby(['sector', 'country', 'year_month']).agg(job_posting_trend=('title', 'count')).reset_index()
    return res

def load_sentiment():
    # FIXED ENCODING HERE
    df = pd.read_csv('data/raw/financial_sentiment.csv', header=None, names=['sentiment', 'sentence'], encoding='latin-1', encoding_errors='replace')
    df['sector'] = df['sentence'].apply(map_sector_by_keywords)
    df = df.dropna(subset=['sector'])
    sentiment_map = {'positive': 1.0, 'neutral': 0.5, 'negative': 0.0}
    df['market_sentiment_score'] = df['sentiment'].map(sentiment_map)
    
    res = df.groupby('sector').agg(market_sentiment_score=('market_sentiment_score', 'mean')).reset_index()
    res['country'] = 'EG'
    res['year_month'] = '2023-01'
    return res

def load_macro(path='data/raw/macro_indicators.csv'):
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='latin-1')

    df.columns = df.columns.str.strip()

    country_map = {
        'Egypt': 'EG',
        'Egypt, Arab Rep.': 'EG',
        'Saudi Arabia': 'SA',
        'United Arab Emirates': 'AE',
        'United States': 'US',
        'United Kingdom': 'GB',
        'Germany': 'DE',
        'France': 'FR',
        'India': 'IN',
        'China': 'CN',
    }

    df['country'] = df['country_name'].map(country_map)
    df = df.dropna(subset=['country'])

    df['year_month'] = df['year'].astype(str) + '-01'

    df = df.rename(columns={
        'Inflation (CPI %)': 'inflation_rate',
        'GDP Growth (% Annual)': 'gdp_growth_pct',
        'Unemployment Rate (%)': 'unemployment_rate',
    })

    # Broadcast to all sectors
    result_rows = []
    for sector in SECTORS:
        temp = df[['country', 'year_month',
                   'inflation_rate']].copy()
        temp['sector'] = sector
        result_rows.append(temp)

    result = pd.concat(result_rows)
    result['inflation_rate'] = pd.to_numeric(
        result['inflation_rate'], errors='coerce'
    ).fillna(0)

    return result[['sector', 'country', 'year_month', 'inflation_rate']]

def load_tech_readiness(path='data/raw/tech_readiness.csv'):
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='latin-1')

    df.columns = df.columns.str.strip()

    # Map full country names to ISO codes
    country_map = {
        'Egypt': 'EG',
        'Saudi Arabia': 'SA',
        'United Arab Emirates': 'AE',
        'United States': 'US',
        'United Kingdom': 'GB',
        'Germany': 'DE',
        'France': 'FR',
        'India': 'IN',
        'China': 'CN',
    }

    df['country'] = df['country'].map(country_map)
    df = df.dropna(subset=['country'])

    df = df.rename(columns={
        'InternetUsers_PctOfPopulationUsingInternet': 'internet_penetration_pct'
    })

    # This dataset has no date — broadcast to all years
    result_rows = []
    for year in range(2010, 2025):
        for sector in SECTORS:
            temp = df[['country', 'internet_penetration_pct']].copy()
            temp['year_month'] = f'{year}-01'
            temp['sector'] = sector
            result_rows.append(temp)

    result = pd.concat(result_rows)
    return result[['sector', 'country', 'year_month', 'internet_penetration_pct']]

def build_feature_matrix():
    print("Building spine...")
    df = build_spine()
    
    loaders = [
        ('Funding Signals', load_funding),
        ('Competitors', load_outcomes),
        ('Job Trends', load_jobs),
        ('Sentiment', load_sentiment),
        ('Macro Economics', load_macro),
        ('Tech Readiness', load_tech_readiness)
    ]
    
    for name, loader in loaders:
        try:
            partial = loader()
            df = df.merge(partial, on=['sector', 'country', 'year_month'], how='left')
            print(f"  [OK] {name} merged.")
        except Exception as e:
            print(f"  [ERROR] {name} failed: {e}")
            
    # Forward fill annual data and handle NaNs
    print("\nCleaning & Scaling...")
    df = df.sort_values(['sector', 'country', 'year_month'])
    feature_cols = [c for c in df.columns if c not in ['sector', 'country', 'year_month']]
    
    df[feature_cols] = df.groupby(['sector', 'country'])[feature_cols].ffill().fillna(0)
    
    # Save unscaled
    df.to_csv('data/X_raw.csv', index=False)
    
    # Scale & Save
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    df_scaled.to_csv('data/X_raw_scaled.csv', index=False)
    
    print(f"\nSUCCESS! Generated {len(df)} rows.")
    print("Saved to data/X_raw.csv and data/X_raw_scaled.csv")

if __name__ == '__main__':
    build_feature_matrix()