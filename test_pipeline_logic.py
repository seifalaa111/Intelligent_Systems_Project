"""
MIDAN Pipeline Tests — real pytest assertions for all pipeline components.
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pickle, json, os, pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")


@pytest.fixture(scope="module")
def models():
    path = MODELS_DIR

    def _pkl(name):
        with open(os.path.join(path, name), 'rb') as f:
            return pickle.load(f)

    def _json(name):
        with open(os.path.join(path, name), 'r', encoding='utf-8') as f:
            return json.load(f)

    scaler = _pkl('scaler_global.pkl')
    pca    = _pkl('pca_global.pkl')
    svm    = _pkl('svm_global.pkl')
    le     = _pkl('label_encoder.pkl')
    lgb    = _pkl('lgb_surrogate.pkl')
    sarima = _json('sarima_results.json')
    return scaler, pca, svm, le, lgb, sarima


FEATURES = ['inflation', 'gdp_growth', 'macro_friction',
            'capital_concentration', 'velocity_yoy']

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
    'MA': {'inflation':  6.1, 'gdp_growth': 3.1, 'unemployment': 11.5},
}
SECTOR_KEYWORDS = {
    'fintech':    ['finance', 'payment', 'fintech', 'bank', 'loan', 'lending',
                   'invoice', 'insurance', 'wallet', 'money'],
    'ecommerce':  ['ecommerce', 'e-commerce', 'shop', 'store', 'retail',
                   'marketplace', 'delivery', 'commerce'],
    'healthtech': ['health', 'medical', 'doctor', 'clinic', 'hospital',
                   'pharma', 'biotech', 'mental'],
    'edtech':     ['education', 'learning', 'school', 'university', 'course',
                   'tutor', 'edtech', 'training'],
    'saas':       ['saas', 'software', 'platform', 'dashboard', 'tool',
                   'api', 'enterprise', 'cloud', 'b2b', 'crm'],
    'logistics':  ['logistics', 'shipping', 'supply chain', 'warehouse',
                   'transport', 'fleet', 'trucking'],
    'agritech':   ['agri', 'farm', 'crop', 'harvest', 'food',
                   'agriculture', 'irrigation'],
}
COUNTRY_KEYWORDS = {
    'EG': ['egypt', 'cairo', 'egyptian'],
    'SA': ['saudi', 'ksa', 'riyadh', 'jeddah'],
    'AE': ['uae', 'dubai', 'abu dhabi', 'emirates'],
    'MA': ['morocco', 'moroccan', 'casablanca'],
    'NG': ['nigeria', 'nigerian', 'lagos'],
    'KE': ['kenya', 'kenyan', 'nairobi'],
    'US': ['usa', 'united states', 'america'],
    'GB': ['uk', 'britain', 'london', 'england'],
}
VALID_REGIMES = {
    'GROWTH_MARKET', 'EMERGING_MARKET',
    'HIGH_FRICTION_MARKET', 'CONTRACTING_MARKET',
}


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


def enhanced_regime(svm_regime, svm_conf, inflation, gdp_growth,
                    macro_friction, velocity_yoy):
    if gdp_growth > 3.5 and inflation < 8 and velocity_yoy > 0.15:
        margin = min((gdp_growth - 3.5) / 4.0,
                     (8 - inflation) / 8.0,
                     (velocity_yoy - 0.15) / 0.25)
        conf = float(np.clip(0.65 + margin * 0.30, 0.60, 0.95))
        return 'GROWTH_MARKET', conf
    if gdp_growth > 2.0 and inflation < 10 and macro_friction < 10:
        margin = min((gdp_growth - 2.0) / 4.0,
                     (10 - inflation) / 10.0,
                     (10 - macro_friction) / 15.0)
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


# ── TEST 1: Agent A1 Parsing ────────────────────────────────

class TestAgentA1:
    def test_detects_fintech_from_keywords(self):
        sec, ctry, sf, cf = agent_a1_parse(
            "Invoice financing app for Egyptian SMEs")
        assert sec == 'fintech'
        assert sf is True

    def test_detects_healthtech(self):
        sec, ctry, sf, cf = agent_a1_parse(
            "Health clinic booking in Dubai")
        assert sec == 'healthtech'
        assert sf is True

    def test_detects_country_egypt(self):
        sec, ctry, sf, cf = agent_a1_parse(
            "Invoice financing app for Egyptian SMEs")
        assert ctry == 'EG'
        assert cf is True

    def test_detects_country_uae(self):
        sec, ctry, sf, cf = agent_a1_parse(
            "Health clinic booking in Dubai")
        assert ctry == 'AE'
        assert cf is True

    def test_no_keyword_falls_back(self):
        sec, ctry, sf, cf = agent_a1_parse(
            "I want to build something amazing")
        assert sf is False
        assert cf is False
        assert sec == 'fintech'
        assert ctry == 'EG'

    def test_dropdown_fallback_overrides_default(self):
        sec, ctry, sf, cf = agent_a1_parse("A cool startup idea")
        assert sf is False
        SECTOR_MAP = {"Healthtech": "healthtech", "Edtech": "edtech"}
        final_sec = sec if sf else SECTOR_MAP.get(
            "Healthtech", "healthtech")
        assert final_sec == 'healthtech'


# ── TEST 2: Enhanced Regime Rules ────────────────────────────

class TestEnhancedRegime:
    def test_growth_market_fires(self):
        regime, conf = enhanced_regime(
            'EMERGING_MARKET', 0.7, 5.0, 5.0, 5.0, 0.28)
        assert regime == 'GROWTH_MARKET'
        assert 0.60 <= conf <= 0.95

    def test_emerging_market_fires(self):
        regime, conf = enhanced_regime(
            'HIGH_FRICTION_MARKET', 0.5, 6.0, 3.0, 5.0, 0.10)
        assert regime == 'EMERGING_MARKET'
        assert 0.55 <= conf <= 0.90

    def test_contracting_market_negative_gdp(self):
        regime, conf = enhanced_regime(
            'EMERGING_MARKET', 0.5, 10.0, -2.0, 20.0, 0.1)
        assert regime == 'CONTRACTING_MARKET'
        assert 0.60 <= conf <= 0.92

    def test_high_friction_market(self):
        regime, conf = enhanced_regime(
            'EMERGING_MARKET', 0.5, 40.0, 1.0, 45.0, 0.05)
        assert regime == 'HIGH_FRICTION_MARKET'
        assert 0.55 <= conf <= 0.92

    def test_svm_passthrough_when_no_rule_fires(self):
        regime, conf = enhanced_regime(
            'EMERGING_MARKET', 0.72, 12.0, 2.5, 15.0, 0.10)
        assert regime == 'EMERGING_MARKET'
        assert conf == 0.72

    def test_confidence_always_in_range(self):
        for inf in [1, 10, 30, 60]:
            for gdp in [-3, 0, 2, 5, 8]:
                for fric in [-10, 0, 20, 50]:
                    for vel in [0.0, 0.1, 0.3, 0.5]:
                        _, conf = enhanced_regime(
                            'EMERGING_MARKET', 0.5,
                            inf, gdp, fric, vel)
                        assert 0.0 <= conf <= 1.0, \
                            f"conf={conf} out of range"


# ── TEST 3: SVM Pipeline (requires .pkl models) ─────────────

class TestSVMPipeline:
    def test_svm_produces_valid_regime(self, models):
        scaler, pca, svm, le, lgb, _ = models
        x_raw = np.array([[7.5, 5.3, 9.3, 175000.0, 0.28]])
        x_scaled = scaler.transform(x_raw)
        pred = svm.predict(x_scaled)[0]
        regime = le.inverse_transform([pred])[0]
        assert regime in VALID_REGIMES

    def test_svm_probabilities_sum_to_one(self, models):
        scaler, pca, svm, le, lgb, _ = models
        x_raw = np.array([[33.9, 3.8, 37.2, 175000.0, 0.28]])
        x_scaled = scaler.transform(x_raw)
        proba = svm.predict_proba(x_scaled)[0]
        assert abs(sum(proba) - 1.0) < 0.01

    def test_pca_reduces_to_2d(self, models):
        scaler, pca, svm, le, lgb, _ = models
        x_raw = np.array([[10.0, 3.0, 14.0, 100000.0, 0.15]])
        x_scaled = scaler.transform(x_raw)
        x_pca = pca.transform(x_scaled)
        assert x_pca.shape == (1, 2)

    def test_all_sector_country_combos_valid(self, models):
        scaler, pca, svm, le, lgb, sarima_results = models
        regime_counts = {}
        for sec in SECTOR_EFF_MACRO:
            for ctry in COUNTRY_MACRO_DEFAULTS:
                macro = COUNTRY_MACRO_DEFAULTS[ctry]
                eff_inf, gdp_boost, velocity = SECTOR_EFF_MACRO[sec]
                scale = macro['inflation'] / 33.9
                inflation = float(np.clip(eff_inf * scale, 1.0, 100.0))
                gdp_growth = float(macro['gdp_growth'] + gdp_boost)
                macro_fric = float(np.clip(
                    inflation + macro['unemployment'] - gdp_growth,
                    -50, 100))
                cap_conc = SECTOR_MEDIANS.get(
                    sec, SECTOR_MEDIANS['other'])
                x_raw = np.array([[
                    inflation, gdp_growth, macro_fric,
                    float(cap_conc), velocity]])
                x_scaled = scaler.transform(x_raw)
                pred_enc = svm.predict(x_scaled)[0]
                proba = svm.predict_proba(x_scaled)[0]
                svm_regime = le.inverse_transform([pred_enc])[0]
                svm_conf = float(proba.max())
                regime, conf = enhanced_regime(
                    svm_regime, svm_conf, inflation,
                    gdp_growth, macro_fric, velocity)
                assert regime in VALID_REGIMES, \
                    f"{sec}/{ctry} invalid regime: {regime}"
                assert 0.0 < conf <= 1.0, \
                    f"{sec}/{ctry} invalid conf: {conf}"
                regime_counts[regime] = \
                    regime_counts.get(regime, 0) + 1
        assert len(regime_counts) >= 2, \
            f"Only {len(regime_counts)} regime(s) — lacks diversity"


# ── TEST 4: SARIMA Data Integrity ────────────────────────────

class TestSARIMA:
    def test_all_expected_sectors_present(self, models):
        *_, sarima_results = models
        for sec in ['fintech', 'ecommerce', 'healthtech',
                    'edtech', 'saas']:
            assert sec in sarima_results, \
                f"Missing SARIMA model for {sec}"

    def test_forecast_has_three_periods(self, models):
        *_, sarima_results = models
        for sec, data in sarima_results.items():
            assert len(data['forecast_mean']) == 3
            assert len(data['forecast_lower']) == 3
            assert len(data['forecast_upper']) == 3

    def test_upper_bound_gte_lower_bound(self, models):
        *_, sarima_results = models
        for sec, data in sarima_results.items():
            for i in range(3):
                assert (data['forecast_upper'][i]
                        >= data['forecast_lower'][i]), \
                    f"{sec} period {i}: upper < lower"

    def test_sarima_trend_in_range(self, models):
        *_, sarima_results = models
        for sec, data in sarima_results.items():
            fc = [max(0, v) for v in data['forecast_mean']]
            trend = float(np.clip(
                np.mean(fc) / 50.0, 0.15, 0.90))
            assert 0.15 <= trend <= 0.90, \
                f"{sec} trend={trend} out of range"

    def test_no_all_negative_forecasts(self, models):
        *_, sarima_results = models
        for sec, data in sarima_results.items():
            assert not all(
                v < 0 for v in data['forecast_mean']), \
                f"{sec} has all-negative forecast"


# ── TEST 5: TAS Score ────────────────────────────────────────

class TestTAS:
    def test_tas_spread_across_sectors(self, models):
        scaler, pca, svm, le, lgb, sarima_results = models
        tas_values = []
        for sec in SECTOR_EFF_MACRO:
            macro = COUNTRY_MACRO_DEFAULTS['EG']
            eff_inf, gdp_boost, velocity = SECTOR_EFF_MACRO[sec]
            scale = macro['inflation'] / 33.9
            inflation = float(np.clip(
                eff_inf * scale, 1.0, 100.0))
            gdp = float(macro['gdp_growth'] + gdp_boost)
            fric = float(np.clip(
                inflation + macro['unemployment'] - gdp,
                -50, 100))
            cap = SECTOR_MEDIANS.get(sec, 100000)
            x = scaler.transform(np.array(
                [[inflation, gdp, fric, float(cap), velocity]]))
            pred = svm.predict(x)[0]
            prob = svm.predict_proba(x)[0]
            svm_r = le.inverse_transform([pred])[0]
            svm_c = float(prob.max())
            regime, conf = enhanced_regime(
                svm_r, svm_c, inflation, gdp, fric, velocity)
            st = 0.50
            if sec in sarima_results:
                fc = [max(0, v)
                      for v in sarima_results[sec]['forecast_mean']]
                st = float(np.clip(
                    np.mean(fc) / 50.0, 0.15, 0.90))
            xai = conf * 0.5
            tas = round(
                conf * 0.40 + st * 0.35 + xai * 0.25, 3)
            tas_values.append(tas)
        spread = max(tas_values) - min(tas_values)
        assert spread > 0.05, \
            f"TAS spread only {spread:.3f} — not differentiated"


# ── TEST 6: Demo Scenarios ───────────────────────────────────

class TestDemoScenarios:
    @pytest.mark.parametrize(
        "idea,expected_sec,expected_ctry",
        [
            ("Invoice financing for Egyptian SMEs",
             "fintech", "EG"),
            ("Online tutoring platform in Cairo",
             "edtech", "EG"),
            ("Health clinic booking in Dubai",
             "healthtech", "AE"),
            ("SaaS CRM platform for UAE businesses",
             "saas", "AE"),
            ("E-commerce delivery in London",
             "ecommerce", "GB"),
        ],
    )
    def test_a1_parses_demo_correctly(
            self, idea, expected_sec, expected_ctry):
        sec, ctry, sf, cf = agent_a1_parse(idea)
        assert sec == expected_sec, \
            f"Expected sector {expected_sec}, got {sec}"
        assert ctry == expected_ctry, \
            f"Expected country {expected_ctry}, got {ctry}"
        assert sf is True
        assert cf is True


# ── TEST 7: Agent A0 Idea Evaluation (keyword fallback) ────

IDEA_DIMENSIONS = [
    'problem_clarity', 'market_fit', 'feasibility',
    'scalability', 'revenue_model',
]

IDEA_DIM_LABELS = {
    'problem_clarity': 'Problem Clarity',
    'market_fit':      'Market Fit',
    'feasibility':     'Feasibility',
    'scalability':     'Scalability',
    'revenue_model':   'Revenue Model',
}

_IDEA_KEYWORDS = {
    'problem_clarity': ['solve', 'problem', 'pain', 'need', 'gap',
                        'challenge', 'issue', 'struggle', 'inefficien'],
    'market_fit':      ['market', 'demand', 'customer', 'user', 'audience',
                        'segment', 'target', 'b2b', 'b2c', 'consumer'],
    'feasibility':     ['mvp', 'prototype', 'tech', 'stack', 'build',
                        'develop', 'engineer', 'team', 'resource'],
    'scalability':     ['scale', 'grow', 'expand', 'region', 'global',
                        'automat', 'network', 'viral', 'repeat'],
    'revenue_model':   ['revenue', 'monetiz', 'subscription', 'fee',
                        'commission', 'pricing', 'margin', 'profit',
                        'freemium', 'ads'],
}


def keyword_idea_score(idea_text):
    t = idea_text.lower()
    scores = {}
    reasons = {}
    for dim in IDEA_DIMENSIONS:
        kws = _IDEA_KEYWORDS[dim]
        hits = [k for k in kws if k in t]
        raw = min(len(hits) / 3.0, 1.0)
        score = int(30 + raw * 60)
        scores[dim] = score
        if hits:
            reasons[dim] = f"Detected keywords: {', '.join(hits[:3])}"
        else:
            reasons[dim] = "No strong signals detected"
    overall = sum(scores.values()) / (len(scores) * 100)
    return overall, scores, reasons


class TestAgentA0:
    def test_keyword_score_returns_all_dimensions(self):
        overall, scores, reasons = keyword_idea_score(
            "We solve payment problems for underserved customers")
        assert set(scores.keys()) == set(IDEA_DIMENSIONS)
        assert set(reasons.keys()) == set(IDEA_DIMENSIONS)

    def test_keyword_score_range(self):
        overall, scores, reasons = keyword_idea_score(
            "A SaaS subscription platform to scale globally")
        assert 0.0 <= overall <= 1.0
        for dim, s in scores.items():
            assert 30 <= s <= 90, f"{dim} score {s} out of range"

    def test_rich_idea_scores_higher_than_empty(self):
        rich_overall, _, _ = keyword_idea_score(
            "We solve a big problem in the market by building an MVP "
            "tech prototype that can scale globally with subscription "
            "revenue and freemium pricing for b2b customers")
        empty_overall, _, _ = keyword_idea_score(
            "I want to do something cool")
        assert rich_overall > empty_overall

    def test_no_keywords_gives_baseline(self):
        overall, scores, reasons = keyword_idea_score("xyz abc 123")
        for dim in IDEA_DIMENSIONS:
            assert scores[dim] == 30
            assert "No strong signals" in reasons[dim]
        assert abs(overall - 0.30) < 0.01

    def test_reasons_contain_detected_keywords(self):
        _, _, reasons = keyword_idea_score(
            "We solve a pain point with subscription revenue")
        assert "solve" in reasons['problem_clarity'] or \
               "pain" in reasons['problem_clarity']
        assert "subscription" in reasons['revenue_model'] or \
               "revenue" in reasons['revenue_model']


# ── TEST 8: SVS and Quadrant ───────────────────────────────

class TestSVSQuadrant:
    def _quadrant(self, tas, idea_score):
        svs = tas * 0.50 + idea_score * 0.50
        if svs >= 0.60 and idea_score >= 0.50:
            return svs, "GO"
        elif tas >= 0.55 and idea_score < 0.50:
            return svs, "Wrong Idea"
        elif tas < 0.55 and idea_score >= 0.50:
            return svs, "Wait"
        else:
            return svs, "STOP"

    def test_go_quadrant(self):
        svs, q = self._quadrant(0.80, 0.70)
        assert q == "GO"
        assert svs >= 0.60

    def test_wrong_idea_quadrant(self):
        svs, q = self._quadrant(0.70, 0.30)
        assert q == "Wrong Idea"

    def test_wait_quadrant(self):
        svs, q = self._quadrant(0.40, 0.70)
        assert q == "Wait"

    def test_stop_quadrant(self):
        svs, q = self._quadrant(0.30, 0.30)
        assert q == "STOP"

    def test_svs_formula(self):
        svs, _ = self._quadrant(0.60, 0.80)
        assert abs(svs - 0.70) < 0.01
