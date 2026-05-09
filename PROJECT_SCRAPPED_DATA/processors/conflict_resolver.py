"""
MIDAN Data Pipeline -- Conflict Resolver & Confidence Engine
Resolves multi-source contradictions and calculates rigid confidence scores.
"""

from collections import defaultdict
import math

class ConflictResolver:
    """
    Groups raw extracted entries for a single startup and intelligently merges them.
    Assigns a unified confidence score based on evidence weight and contradiction penalties.
    """

    def __init__(self):
        self.scalar_fields = [
            "primary_industry", "secondary_industry", "business_model", "target_user", "target_segment",
            "value_proposition", "differentiation",
            "switching_cost", "competition_density",
            "retention_proxy", "onboarding_friction",
            "monetization_strength", "competition_intensity",
            "funding_stage", "market_context",
        ]
        self.list_fields = [
            "pain_points", "adoption_barriers", "user_complaints",
            "success_drivers", "failure_reasons"
        ]

    def resolve(self, entries: list[dict]) -> dict:
        """Resolve a group of entries for a single startup."""
        if not entries:
            return {}

        merged = entries[0].copy()
        
        sources_used = list(set(e.get("source_type", "") for e in entries if e.get("source_type")))
        merged["_sources"] = sources_used
        
        all_conflicts = []
        global_confidence_penalties = 0

        # Resolve Scalar Fields & L3 Signals
        for field in self.scalar_fields:
            votes = defaultdict(list)  # value -> list of sources
            evidence_vault = defaultdict(list) # value -> accumulated evidence snippets
            
            for entry in entries:
                val = entry.get(field, "").strip()
                if val:
                    source = entry.get("source_type", "unknown")
                    votes[val].append(source)
                    
                    # Store evidence if available
                    signal_ev = entry.get("signal_evidence", {}).get(field, [])
                    if signal_ev:
                        evidence_vault[val].extend(signal_ev)

            if not votes:
                merged[field] = ""
                continue

            # Check for conflict
            distinct_values = list(votes.keys())
            
            if len(distinct_values) == 1:
                # Unanimous agreement
                winner = distinct_values[0]
                merged[field] = winner
                if winner in evidence_vault:
                    merged.setdefault("signal_evidence", {})[field] = list(dict.fromkeys(evidence_vault[winner]))[:3]
                
            else:
                # CONFLICT DETECTED
                # Sort by number of votes
                sorted_votes = sorted(votes.items(), key=lambda x: len(x[1]), reverse=True)
                winner, winner_sources = sorted_votes[0]
                runner_up, runner_sources = sorted_votes[1]

                # Hard vs Soft Conflict
                # Enums (high/medium/low) count as HARD conflicts. String text counts as SOFT.
                is_enum = field in ["retention_proxy", "onboarding_friction", "monetization_strength", "competition_intensity", "switching_cost"]
                if is_enum:
                    all_conflicts.append(f"HARD CONFLICT in {field}: {winner} ({len(winner_sources)} votes) vs {runner_up} ({len(runner_sources)} votes)")
                    global_confidence_penalties += 0.3
                else:
                    all_conflicts.append(f"SOFT CONFLICT in {field}: Strings differed across sources.")
                    global_confidence_penalties += 0.1

                merged[field] = winner
                if winner in evidence_vault:
                    merged.setdefault("signal_evidence", {})[field] = list(dict.fromkeys(evidence_vault[winner]))[:3]

        # Process List Fields (Union)
        for field in self.list_fields:
            combined = []
            seen = set()
            for entry in entries:
                for item in entry.get(field, []):
                    cleaned = str(item).strip()
                    if cleaned and cleaned.lower() not in seen:
                        seen.add(cleaned.lower())
                        combined.append(cleaned)
            merged[field] = combined

        # Final Confidence Calibration
        base_confidence = self._calculate_confidence(merged, len(entries))
        final_confidence = max(0.0, base_confidence - global_confidence_penalties)
        merged["confidence_score"] = round(final_confidence, 2)
        
        if all_conflicts:
            merged["conflict_notes"] = all_conflicts

        return merged

    def _calculate_confidence(self, entry: dict, exact_run_count: int) -> float:
        """
        Calculate weighted confidence natively scaled [0,1].
        Weights explicitly judge snippet textual strength vs weak words.
        """
        evidence = entry.get("signal_evidence", {})
        weak_phrases = ["may", "might", "potentially", "probably", "could", "seems", "likely", "maybe"]
        
        weighted_snippets = 0.0
        
        for snips in evidence.values():
            for text in snips:
                t_lower = text.lower()
                if any(w in t_lower for w in weak_phrases):
                    weighted_snippets += 0.3  # Weak inference
                else:
                    weighted_snippets += 1.0  # Explicit statement
        
        # Max out quality at 4 explicit snippets
        evidence_score = min(1.0, weighted_snippets / 4.0) * 0.5
        
        # Sources modifier
        unique_sources = len(entry.get("_sources", []))
        source_score = min(1.0, unique_sources / 3.0) * 0.3
        
        # Baseline consistency
        consistency_score = 0.2

        raw_total = evidence_score + source_score + consistency_score
        return raw_total
