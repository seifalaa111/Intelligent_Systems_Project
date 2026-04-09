"""
MIDAN POST-FIX VERIFICATION — Tests the fixed pipeline logic
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pickle, json, os, sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
path = MODELS_DIR

scaler = pickle.load(open(f'{path}/scaler_global.pkl','rb'))
pca    = pickle.load(open(f'{path}/pca_global.pkl','rb'))
svm    = pickle.load(open(f'{path}/svm_global.pkl','rb'))
le     = pickle.load(open(f'{path}/label_encoder.pkl','rb'))
lgb    = pickle.load(open(f'{path}/lgb_surrogate.pkl','rb'))
sarima_results = json.load(open(f'{path}/sarima_results.json'))

FEATURES = ['inflation','gdp_growth','macro_friction','capital_concentration','velocity_yoy']
SECTOR_MEDIANS = {
    'fintech': 175000.0, 'ecommerce': 120000.0, 'healthtech': 200000.0,
    'edtech': 80000.0, 'saas': 250000.0, 'logistics': 90000.0,
    'agritech': 50000.0, 'other': 100000.0,
}
SECTOR_EFF_MACRO = {
    'fintech':    (7.5,  +1.5, 0.28), 'healthtech': (7.0,  +2.0, 0.22),
    'saas':       (4.0,  +2.2, 0.10), 'agritech':   (4.5,  +0.7, 0.12),
    'edtech':     (40.0, -1.0, 0.07), 'logistics':  (42.0, -1.8, 0.09),
    'ecommerce':  (36.0, -1.3, 0.13), 'other':      (33.9,  0.0, 0.10),
}
COUNTRY_MACRO_DEFAULTS = {
    'EG': {'inflation': 33.9, 'gdp_growth': 3.8, 'unemployment': 7.1},
    'SA': {'inflation':  2.3, 'gdp_growth': 1.9, 'unemployment': 6.1},
    'AE': {'inflation':  1.6, 'gdp_growth': 4.2, 'unemployment': 3.1},
    'US': {'inflation':  3.4, 'gdp_growth': 2.5, 'unemployment': 3.7},
    'GB': {'inflation':  4.0, 'gdp_growth': 0.1, 'unemployment': 4.2},
    'NG': {'inflation': 28.9, 'gdp_growth': 3.3, 'unemployment': 4.1},
    'KE': {'inflation':  6.3, 'gdp_growth': 5.6, 'unemployment': 5.7},
    'MA': {'inflation':  6.1, 'gdp_growth': 3.1, 'unemployment':11.5},
}
SECTOR_KEYWORDS = {
    'fintech':    ['finance','payment','fintech','bank','loan','lending',
                   'invoice','insurance','wallet','money'],
    'ecommerce':  ['ecommerce','e-commerce','shop','store','retail',
                   'marketplace','delivery','commerce'],
    'healthtech': ['health','medical','doctor','clinic','hospital',
                   'pharma','biotech','mental'],
    'edtech':     ['education','learning','school','university','course',
                   'tutor','edtech','training'],
    'saas':       ['saas','software','platform','dashboard','tool',
                   'api','enterprise','cloud','b2b','crm'],
    'logistics':  ['logistics','shipping','supply chain','warehouse',
                   'transport','fleet','trucking'],
    'agritech':   ['agri','farm','crop','harvest','food',
                   'agriculture','irrigation'],
}
COUNTRY_KEYWORDS = {
    'EG': ['egypt','cairo','egyptian'], 'SA': ['saudi','ksa','riyadh','jeddah'],
    'AE': ['uae','dubai','abu dhabi','emirates'], 'MA': ['morocco','moroccan','casablanca'],
    'NG': ['nigeria','nigerian','lagos'], 'KE': ['kenya','kenyan','nairobi'],
    'US': ['usa','united states','america'], 'GB': ['uk','britain','london','england'],
}

# FIXED agent_a1_parse
def agent_a1_parse(idea_text):
    t = idea_text.lower()
    sector, sector_found = None, False
    for sec, kws in SECTOR_KEYWORDS.items():
        if any(k in t for k in kws):
            sector, sector_found = sec, True
            break
    if not sector_found:
        sector = 'fintech'
    country, country_found = None, False
    for code, kws in COUNTRY_KEYWORDS.items():
        if any(k in t for k in kws):
            country, country_found = code, True
            break
    if not country_found:
        country = 'EG'
    return sector, country, sector_found, country_found

# FIXED enhanced_regime
def enhanced_regime(svm_regime, svm_conf, inflation, gdp_growth, macro_friction, velocity_yoy):
    if gdp_growth > 3.5 and inflation < 8 and velocity_yoy > 0.15:
        margin = min((gdp_growth-3.5)/4.0, (8-inflation)/8.0, (velocity_yoy-0.15)/0.25)
        conf = float(np.clip(0.65 + margin * 0.30, 0.60, 0.95))
        return 'GROWTH_MARKET', conf
    if gdp_growth > 2.0 and inflation < 10 and macro_friction < 10:
        margin = min((gdp_growth-2.0)/4.0, (10-inflation)/10.0, (10-macro_friction)/15.0)
        conf = float(np.clip(0.60 + margin * 0.30, 0.55, 0.90))
        return 'EMERGING_MARKET', conf
    if gdp_growth < 0 or (inflation > 50 and macro_friction > 50):
        severity = max(abs(min(gdp_growth, 0)) / 3.0, 0.0)
        conf = float(np.clip(0.65 + severity * 0.25, 0.60, 0.92))
        return 'CONTRACTING_MARKET', conf
    if macro_friction > 30 or inflation > 25:
        pain = max((macro_friction - 30) / 40, (inflation - 25) / 30, 0)
        conf = float(np.clip(0.60 + pain * 0.30, 0.55, 0.92))
        return 'HIGH_FRICTION_MARKET', conf
    return svm_regime, svm_conf

print("=" * 90)
print("MIDAN POST-FIX VERIFICATION")
print("=" * 90)

# TEST 1: Agent A1 dropdown fallback
print("\n--- TEST 1: AGENT A1 DROPDOWN FALLBACK ---")
tests = [
    ("I want to build something amazing", "Healthtech", "AE"),
    ("Invoice financing app for Egyptian SMEs", "Other", "SA"),
    ("A cool startup idea", "Edtech", "SA"),
    ("Health clinic booking in Dubai", "Fintech", "EG"),
]
SECTOR_MAP = {"Healthtech":"healthtech","Edtech":"edtech","Fintech":"fintech","Other":"other"}

print(f"{'Input':<42} {'Dropdown':<20} {'Result':<20} {'Source'}")
print("-" * 100)
for text, dd_sec, dd_ctry in tests:
    ps, pc, sf, cf = agent_a1_parse(text)
    final_sec = ps if sf else SECTOR_MAP.get(dd_sec, dd_sec.lower())
    final_ctry = pc if cf else dd_ctry
    src_s = "A1 text" if sf else "DROPDOWN"
    src_c = "A1 text" if cf else "DROPDOWN"
    print(f"{text:<42} {dd_sec+'/'+dd_ctry:<20} {final_sec+'/'+final_ctry:<20} sec={src_s}, ctry={src_c}")

# TEST 2: Regime distribution
print("\n--- TEST 2: REGIME RESULTS FOR ALL SECTOR x COUNTRY ---")
sectors = list(SECTOR_EFF_MACRO.keys())
countries = list(COUNTRY_MACRO_DEFAULTS.keys())

regime_counts = {}
results_table = []

for sec in sectors:
    for ctry in countries:
        macro = COUNTRY_MACRO_DEFAULTS[ctry]
        eff_inf, gdp_boost, velocity = SECTOR_EFF_MACRO[sec]
        scale = macro['inflation'] / 33.9
        inflation = float(np.clip(eff_inf * scale, 1.0, 100.0))
        gdp_growth = float(macro['gdp_growth'] + gdp_boost)
        macro_fric = float(np.clip(inflation + macro['unemployment'] - gdp_growth, -50, 100))
        cap_conc = SECTOR_MEDIANS.get(sec, SECTOR_MEDIANS['other'])

        x_raw = np.array([[inflation, gdp_growth, macro_fric, float(cap_conc), velocity]])
        x_scaled = scaler.transform(x_raw)
        pred_enc = svm.predict(x_scaled)[0]
        proba = svm.predict_proba(x_scaled)[0]
        svm_regime = le.inverse_transform([pred_enc])[0]
        svm_conf = float(proba.max())
        regime, conf = enhanced_regime(svm_regime, svm_conf, inflation, gdp_growth, macro_fric, velocity)

        # SARIMA
        sarima_trend = 0.50
        if sec in sarima_results:
            fc = [max(0, v) for v in sarima_results[sec]['forecast_mean']]
            fc_mean = float(np.mean(fc))
            sarima_trend = float(np.clip(fc_mean / 50.0, 0.15, 0.90))

        xai_score = conf * 0.5  # approx
        tas = round(conf * 0.40 + sarima_trend * 0.35 + xai_score * 0.25, 3)
        action = tas >= 0.70 and regime in ('GROWTH_MARKET', 'EMERGING_MARKET')

        regime_counts[regime] = regime_counts.get(regime, 0) + 1
        results_table.append((sec, ctry, regime, conf, sarima_trend, tas, action))

# Print Egypt results
print(f"\n{'Sector':<12} {'Ctry':<5} {'Regime':<22} {'Conf':>6} {'SARIMA':>7} {'TAS':>6} {'Action'}")
print("-" * 75)
for sec, ctry, regime, conf, sarima, tas, action in results_table:
    if ctry in ('EG', 'AE', 'SA', 'GB'):
        mark = ">> FIRE" if action else ""
        print(f"{sec:<12} {ctry:<5} {regime:<22} {conf:>5.0%} {sarima:>7.2f} {tas:>6.3f} {mark}")

# TEST 3: Regime distribution
print(f"\n--- TEST 3: REGIME DISTRIBUTION (64 combos) ---")
for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
    print(f"   {regime:<25} {count:>3} ({count/64:.0%})")

# TEST 4: TAS spread
all_tas = [t[5] for t in results_table]
print(f"\n--- TEST 4: TAS SCORE SPREAD ---")
print(f"   Min TAS:  {min(all_tas):.3f}")
print(f"   Max TAS:  {max(all_tas):.3f}")
print(f"   Spread:   {max(all_tas)-min(all_tas):.3f}")
print(f"   Mean:     {np.mean(all_tas):.3f}")
print(f"   Actions fired: {sum(1 for t in results_table if t[6])}")

# TEST 5: SARIMA diversity
print(f"\n--- TEST 5: SARIMA TREND VALUES (should NOT be binary) ---")
for sec in ['fintech','ecommerce','healthtech','edtech','saas','logistics','agritech','other']:
    if sec in sarima_results:
        fc = [max(0, v) for v in sarima_results[sec]['forecast_mean']]
        fc_mean = float(np.mean(fc))
        trend = float(np.clip(fc_mean / 50.0, 0.15, 0.90))
        print(f"   {sec:<12} mean={fc_mean:>6.1f} -> trend={trend:.2f}")
    else:
        print(f"   {sec:<12} NO MODEL -> trend=0.50")

# TEST 6: Key demo scenarios
print(f"\n--- TEST 6: DEMO SCENARIOS ---")
demos = [
    ("Invoice financing for Egyptian SMEs", "fintech", "EG"),
    ("Online tutoring platform in Cairo", "edtech", "EG"),
    ("Health clinic booking in Dubai", "healthtech", "AE"),
    ("SaaS CRM platform for UAE businesses", "saas", "AE"),
    ("E-commerce delivery in London", "ecommerce", "GB"),
    ("Logistics fleet management in Saudi", "logistics", "SA"),
]
print(f"{'Idea':<45} {'Sector':<12} {'Ctry':<5} {'Regime':<22} {'TAS':>6} {'Verdict'}")
print("-" * 100)
for idea, exp_sec, exp_ctry in demos:
    ps, pc, sf, cf = agent_a1_parse(idea)
    sec = ps if sf else exp_sec
    ctry = pc if cf else exp_ctry
    macro = COUNTRY_MACRO_DEFAULTS[ctry]
    eff_inf, gdp_boost, velocity = SECTOR_EFF_MACRO[sec]
    scale = macro['inflation'] / 33.9
    inflation = float(np.clip(eff_inf * scale, 1.0, 100.0))
    gdp = float(macro['gdp_growth'] + gdp_boost)
    fric = float(np.clip(inflation + macro['unemployment'] - gdp, -50, 100))
    cap = SECTOR_MEDIANS.get(sec, 100000)
    x = scaler.transform(np.array([[inflation, gdp, fric, float(cap), velocity]]))
    pred = svm.predict(x)[0]; prob = svm.predict_proba(x)[0]
    svm_r = le.inverse_transform([pred])[0]; svm_c = float(prob.max())
    regime, conf = enhanced_regime(svm_r, svm_c, inflation, gdp, fric, velocity)
    st = 0.50
    if sec in sarima_results:
        fc = [max(0,v) for v in sarima_results[sec]['forecast_mean']]
        st = float(np.clip(np.mean(fc)/50.0, 0.15, 0.90))
    xai = conf * 0.5
    tas = round(conf*0.40 + st*0.35 + xai*0.25, 3)
    fired = tas >= 0.70 and regime in ('GROWTH_MARKET','EMERGING_MARKET')
    verdict = "GO - WEBHOOK FIRED" if fired else ("CAUTION" if tas >= 0.50 else "STOP")
    print(f"{idea:<45} {sec:<12} {ctry:<5} {regime:<22} {tas:>6.3f} {verdict}")

print("\n" + "=" * 90)
print("VERIFICATION COMPLETE")
print("=" * 90)
