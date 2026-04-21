"""
MIDAN Data Pipeline -- Phase 4 Decision Engine
Implements probabilistic evaluation matrix. 
Generates continuous scoring [-1, 1], secondary risk hierarchies, and logic strings derived explicitly from grounded signals.
"""

from collections import defaultdict
from datetime import datetime

class DecisionEngine:
    """
    Transforms extracted intelligence into contextual decisions under uncertainty.
    """

    def evaluate_payload(self, entries: list[dict]) -> list[dict]:
        processed = []
        for e in entries:
            processed.append(self._evaluate_single(e))
        return processed

    def _evaluate_single(self, entry: dict) -> dict:
        score = 0.0
        
        # 1. Base Parameter Assignment
        model = entry.get("business_model", "").lower()
        industry = entry.get("primary_industry", "").lower()
        target = entry.get("target_segment", "").lower()
        comp = entry.get("competition_intensity", "")
        retention = entry.get("retention_proxy", "")
        monetization = entry.get("monetization_strength", "")
        friction = entry.get("onboarding_friction", "")
        
        def cap_influence(val):
            return max(-0.35, min(0.35, val))

        # 2. Probabilistic Rules Engine (Mathematical Evaluation)
        comp_penalty = cap_influence(-0.35 if comp == "high" else 0.0)
        margin_bonus = cap_influence(0.35 if monetization == "strong" else 0.0)
        retention_bonus = cap_influence(0.35 if retention == "high" else -0.3)
        friction_penalty = cap_influence(-0.35 if friction == "high" else 0.2)
        
        # Trade-off 1: Margin & Retention cancels Heavy Competition
        if comp == "high" and margin_bonus > 0 and retention_bonus > 0:
            comp_penalty = 0.0  
            
        # Trade-off 2: Hardware Penalty Scaling
        hardware_penalty = 0.0
        if "hardware" in model or "hardware" in industry:
            # Amplified heavily if competition is high
            comp_multiplier = 0.8 if comp == "high" else 0.3
            hardware_penalty = cap_influence(-0.6 * comp_multiplier)
            
        # Trade-off 3: B2C Friction Dropoff Limits
        if friction == "high" and "b2c" in target:
            friction_penalty = cap_influence(friction_penalty * 1.5)
            
        score = sum([comp_penalty, margin_bonus, retention_bonus, friction_penalty, hardware_penalty])
        score = max(-1.0, min(1.0, score))
        
        # 3. Confidence Propagation
        sig_conf = entry.get("confidence_score", 0.0)
        p_conf = entry.get("pattern_confidence_score", 0.0)
        con_penalty = 0.2 if entry.get("conflict_notes") else 0.0
        
        final_confidence = (
            sig_conf * 0.5 + 
            p_conf * 0.3 + 
            (1.0 - con_penalty) * 0.2
        )
        
        # Traceability Registry Array
        trace = {
            "score_components": {
                "comp_penalty": comp_penalty,
                "margin_bonus": margin_bonus,
                "retention_bonus": retention_bonus,
                "friction_penalty": friction_penalty,
                "hardware_penalty": hardware_penalty
            },
            "signals_used": [k for k, v in entry.items() if k in ["primary_industry", "business_model", "target_segment", "competition_intensity", "retention_proxy", "monetization_strength", "onboarding_friction"] and v],
            "patterns_used": [],
            "conflicts_detected": entry.get("conflict_notes", [])
        }

        # 4. Enforce Pattern Threshold Limits (freq >= 3 + meaningful variance)
        valid_pats = []
        highest_fail_rate = 0.0
        highest_fail_tag = None
        
        for p in entry.get("system_patterns", []):
            freq = entry.get("pattern_frequencies", {}).get(p, 0)
            fr = float(entry.get("pattern_failure_rates", {}).get(p, 0.0))
            is_polarized = abs(fr - 0.5) >= 0.2  # Meaningful if fail rate > 70% or < 30%
            
            if freq >= 3 and p_conf >= 0.15 and is_polarized: 
                valid_pats.append(p)
                trace["patterns_used"].append(p)
                if fr > highest_fail_rate:
                    highest_fail_rate = fr
                    highest_fail_tag = p

        # 5. Core Decision State Generation
        decision = "CONDITIONAL"
        if score >= 0.25:
            decision = "GO"
        elif score <= -0.25:
            decision = "NO-GO"
            
        # Hard Guardrails & Fallbacks
        if len(trace["signals_used"]) < 3 or final_confidence < 0.3:
            decision = "CONDITIONAL"
            score = 0.0
            primary_reason = "Insufficient extracted signal footprint enforcing a baseline logic collapse. Reverting to CONDITIONAL pending required verification boundaries."
        elif final_confidence < 0.5 and decision == "GO":
            decision = "CONDITIONAL"
            primary_reason = "Evaluated logic array natively maps a GO, but weak confidence metrics explicitly bar full structural execution. Hard fallback engaged."
        else:
            # 6. Structurally Built Reasoning
            primary_reason = ""
            if valid_pats and highest_fail_tag:
                fail_p = int(highest_fail_rate * 100)
                status_text = "historic baseline survival"
                if fail_p >= 50:
                    status_text = f"historic structural failure rate of {fail_p}%"
                    
                primary_reason = f"Extracted metrics align strongly with [{highest_fail_tag}] forcing a {status_text}."
                if decision == "NO-GO":
                    primary_reason += f" Weighted structural metrics resulted in a {round(score, 2)} decay score rendering lifecycle execution unviable."
                elif decision == "GO":
                    primary_reason += f" Despite generic pattern hazards, intrinsic structural advantages correctly resolved into a [{round(score, 2)}] deterministic baseline."
            else:
                if decision == "GO":
                     primary_reason = f"Unit economics mathematically balance [{round(score, 2)}], signaling strong defensible moat characteristics independently capable of nullifying downstream churn."
                elif decision == "NO-GO":
                     primary_reason = f"Fatal negative divergence [{round(score, 2)}] within foundational architecture limits standard operating expansion completely."
                else:
                     primary_reason = f"Weighted pipeline evaluated precisely at {round(score, 2)}, establishing neither massive systemic adoption nor immediate pipeline implosion."

        # 7. Surrogate Risk Taxonomies
        risks = []
        req_changes = []
        
        if friction == "high":
            risks.append({"dominant": "onboarding_friction (high)"})
            risks.append({"second_order": "Rapid initial lifecycle drop-off resulting in elevated CAC-payback thresholds.", "impact": "high"})
            req_changes.append("Implement zero-auth usage funnels isolating friction exclusively to end-cycle monetization nodes.")
            
        if retention == "low":
            risks.append({"dominant": "retention_proxy (low)"})
            risks.append({"second_order": "Systemic depletion of active user networks driving total scale insolvency.", "impact": "critical"})
            req_changes.append("Restructure native interaction loops to establish rigid daily/weekly dependencies.")

        if comp == "high" and monetization != "strong":
            risks.append({"dominant": "competition_intensity (high) vs margin deficit"})
            risks.append({"second_order": "Inability to out-acquire established market incumbents.", "impact": "high"})
            req_changes.append("Hollow out generalized market offering to deeply serve hyper-niche, currently unmonetized silos.")

        if hardware_penalty < 0:
            risks.append({"dominant": "hardware deployment inertia"})
            risks.append({"second_order": "Total capital depletion prior to validating foundational iteration loops.", "impact": "critical"})
            req_changes.append("Bypass localized manufacturing limits and abstract pure software overlay utilizing third-party physical integrations.")

        # Simplified Executive Summary
        user_summary = f"System recommends a {decision} decision."
        if decision == "NO-GO":
            user_summary += " Severe structural vulnerabilities exist making scale unviable."
        elif decision == "CONDITIONAL":
            user_summary += " Validation is incomplete or critical pivot requirements exist."
        elif decision == "GO":
            user_summary += " Economics map perfectly against historic success patterns."

        entry["decision_analysis"] = {
            "engine_version": "v1.0",
            "evaluated_at": datetime.now().isoformat(),
            "decision": decision,
            "decision_confidence": round(final_confidence, 2),
            "internal_logic_score": round(score, 2),
            "primary_reason": primary_reason,
            "user_summary": user_summary,
            "key_risks": risks,
            "required_changes": req_changes,
            "decision_trace": trace,
            "user_override": {
                "decision": "",
                "reason": "",
                "override_timestamp": ""
            }
        }
        
        return entry
