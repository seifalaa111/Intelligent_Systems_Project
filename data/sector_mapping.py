import pandas as pd

SECTORS = ['fintech', 'healthtech', 'edtech', 'ecommerce', 'logistics', 'saas', 'agritech', 'proptech', 'cleantech']
COUNTRIES = ['EG', 'SA', 'AE', 'US', 'GB']

# Keyword matcher for text columns (like job titles and news sentences)
def map_sector_by_keywords(text):
    if pd.isna(text): return None
    text = str(text).lower()
    
    keywords = {
        'fintech': ['fintech', 'finance', 'bank', 'payment', 'crypto', 'pay', 'wealth'],
        'healthtech': ['health', 'medical', 'pharma', 'clinic', 'doctor', 'patient'],
        'edtech': ['education', 'teacher', 'school', 'learning', 'student', 'university'],
        'ecommerce': ['ecommerce', 'retail', 'shop', 'store', 'marketplace', 'commerce'],
        'logistics': ['logistics', 'supply', 'delivery', 'warehouse', 'freight', 'shipping'],
        'saas': ['software', 'saas', 'cloud', 'developer', 'platform', 'engineer'],
        'agritech': ['agriculture', 'farm', 'crop', 'food'],
        'proptech': ['real estate', 'property', 'housing', 'mortgage', 'broker'],
        'cleantech': ['clean', 'energy', 'solar', 'wind', 'sustainability', 'green']
    }
    
    for sector, kw_list in keywords.items():
        if any(kw in text for kw in kw_list):
            return sector
    return None

def standardize_columns(df):
    # Strips hidden spaces and makes everything lowercase
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df
