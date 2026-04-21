"""
MIDAN Data Pipeline -- Pattern Analyzer (Phase 3)
Extracts combinatorial behavioral signals and enforces cross-dataset validation.
Patterns are only retained if observed across >= 2 startups.
"""

from collections import defaultdict

class PatternAnalyzer:
    """
    Analyzes finalized intelligence entries to extract combinatorial behavioral tags
    representing market dynamics (e.g. low_switching_cost_saas).
    Enforces a global occurrence requirement to prevent assigning hallucinatory tags.
    """

    def __init__(self):
        # A list of matching rule functions. 
        # lambda entry -> pattern_tag (or None)
        self.rules = [
            lambda e: "low_switching_cost_saas" 
                if e.get("business_model") == "SaaS" and e.get("switching_cost") == "low" else None,
            
            lambda e: "high_competition_marketplace" 
                if "Marketplace" in (e.get("business_model", ""), e.get("primary_industry", "")) and e.get("competition_intensity") == "high" else None,

            lambda e: "high_churn_competitive_market"
                if e.get("retention_proxy") == "low" and e.get("competition_intensity") == "high" else None,

            lambda e: "bottlenecked_adoption"
                if e.get("onboarding_friction") == "high" and "Enterprise" not in e.get("primary_industry", "") else None,

            lambda e: "strong_pricing_power"
                if e.get("monetization_strength") == "strong" and e.get("retention_proxy") == "high" else None,

            lambda e: "hard_to_monetize_social"
                if e.get("primary_industry") == "Social" and e.get("monetization_strength") in ("weak", "") else None,
                
            lambda e: "developer_reliant"
                if "Dev" in e.get("primary_industry", "") and e.get("onboarding_friction") == "low" else None,
                
            lambda e: "hardware_margin_trap"
                if "Hardware" in (e.get("primary_industry", ""), e.get("business_model", "")) and e.get("competition_intensity") == "high" else None
        ]

    def analyze_patterns(self, entries: list[dict]) -> list[dict]:
        """Process all entries, assign patterns, compute correlations."""
        # 1. Map candidate patterns
        pattern_frequency = defaultdict(int)
        
        # Add success/failure metric maps
        # map: pattern -> {'success': 0, 'fail': 0}
        pattern_correlations = defaultdict(lambda: {'success': 0, 'fail': 0})
        
        for entry in entries:
            entry["system_patterns"] = []
            
            # Simple heuristic for outcome
            is_failure = bool(entry.get("failure_reasons", []))
            
            for rule in self.rules:
                tag = rule(entry)
                if tag:
                    entry["system_patterns"].append(tag)
                    pattern_frequency[tag] += 1
                    if is_failure:
                        pattern_correlations[tag]['fail'] += 1
                    else:
                        pattern_correlations[tag]['success'] += 1
                    
        # 2. Enforce Frequency Rule (must appear in >= 2 startups)
        valid_patterns = {tag for tag, count in pattern_frequency.items() if count >= 2}
        
        # Precompute Rates for valid patterns
        valid_rates = {}
        for tag in valid_patterns:
            total = pattern_correlations[tag]['success'] + pattern_correlations[tag]['fail']
            valid_rates[tag] = {
                "success_rate": round(pattern_correlations[tag]['success'] / total, 2),
                "failure_rate": round(pattern_correlations[tag]['fail'] / total, 2)
            }
        
        # 3. Clean and Score
        for entry in entries:
            final_patterns = []
            pattern_success_rates = {}
            pattern_failure_rates = {}
            pattern_frequencies = {}
            
            for tag in entry.get("system_patterns", []):
                if tag in valid_patterns:
                    final_patterns.append(tag)
                    pattern_success_rates[tag] = valid_rates[tag]["success_rate"]
                    pattern_failure_rates[tag] = valid_rates[tag]["failure_rate"]
                    pattern_frequencies[tag] = pattern_frequency[tag]
                    
            entry["system_patterns"] = final_patterns
            entry["pattern_success_rates"] = pattern_success_rates
            entry["pattern_failure_rates"] = pattern_failure_rates
            entry["pattern_frequencies"] = pattern_frequencies
            
            # Additional confidence bump if patterns are detected naturally
            if len(final_patterns) > 0:
                p_conf = round(min(1.0, len(final_patterns) * 0.15), 2)
                entry["pattern_confidence_score"] = p_conf
            else:
                entry["pattern_confidence_score"] = 0.0

        return entries
