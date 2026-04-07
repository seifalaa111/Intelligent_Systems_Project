import pandas as pd
import os

files = {
    'funding_2024':         'data/raw/funding_2024.csv',
    'crunchbase_outcomes':  'data/raw/crunchbase_outcomes.csv',
    'linkedin_jobs':        'data/raw/linkedin_jobs.csv',
    'financial_sentiment':  'data/raw/financial_sentiment.csv',
    'gii_scores':           'data/raw/gii_scores.csv',
    'doing_business':       'data/raw/doing_business.csv',
    'macro_indicators':     'data/raw/macro_indicators.csv',
    'ecommerce_demand':     'data/raw/ecommerce_demand.csv',
    'unicorns':             'data/raw/unicorns.csv',
    'tech_readiness':       'data/raw/tech_readiness.csv',
    'macro_timeseries':     'data/raw/macro_timeseries.csv',
}

for name, path in files.items():
    print(f"\n{'='*60}")
    print(f"DATASET: {name}")
    print(f"FILE: {path}")
    print(f"{'='*60}")
    
    if not os.path.exists(path):
        print("ERROR: FILE NOT FOUND")
        continue
        
    size_kb = os.path.getsize(path) / 1024
    print(f"Size: {size_kb:.1f} KB")
    
    try:
        df = pd.read_csv(path, nrows=3, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, nrows=3, encoding='latin-1')
        except Exception as e:
            print(f"ERROR reading file: {e}")
            continue
    except Exception as e:
        print(f"ERROR reading file: {e}")
        continue
        
    print(f"Columns ({len(df.columns)}):")
    for col in df.columns.tolist():
        print(f"  - {col}")
    
    print(f"\nShape (first 3 rows): {df.shape}")
    print(f"\nNull counts:")
    print(df.isnull().sum().to_dict())
