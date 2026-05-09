"""Phase 2 validation -- full classification and signal quality check."""
import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open("data/structured/startup_intelligence.json", encoding="utf-8") as f:
    data = json.load(f)

total = len(data["startups"])
print(f"Pipeline v{data['metadata']['pipeline_version']}")
print(f"Total entries: {total}")
print(f"Sources: {data['metadata']['sources_used']}")
print()

# === CLASSIFICATION TABLE ===
print(f"{'Startup':<20} | {'Industry':<22} | {'Model':<12} | {'Seg':<10} | {'SW':<6} | {'Ret':<6} | {'Onb':<6} | {'Mon':<8} | {'Comp':<6}")
print("-" * 120)

for s in data["startups"]:
    print(f"{s['startup_name'][:20]:<20} | "
          f"{s.get('industry','')[:22]:<22} | "
          f"{s.get('business_model','')[:12]:<12} | "
          f"{s.get('target_segment','')[:10]:<10} | "
          f"{s.get('switching_cost','')[:6]:<6} | "
          f"{s.get('retention_proxy','')[:6]:<6} | "
          f"{s.get('onboarding_friction','')[:6]:<6} | "
          f"{s.get('monetization_strength','')[:8]:<8} | "
          f"{s.get('competition_intensity','')[:6]:<6}")

# === PHASE 1 vs PHASE 2 COMPARISON ===
print("\n" + "=" * 60)
print("CLASSIFICATION FIX VALIDATION")
print("=" * 60)

fixes = {
    "figma": {"phase1_model": "ADTECH", "phase1_industry": "Enterprise Software"},
    "stripe": {"phase1_model": "SAAS", "phase1_industry": "Devtools"},
    "loom": {"phase1_model": "EDTECH", "phase1_industry": "Enterprise Software"},
}

for s in data["startups"]:
    name = s["startup_name"].lower()
    if name in fixes:
        p1 = fixes[name]
        print(f"\n  {s['startup_name']}:")
        print(f"    Phase 1: model={p1['phase1_model']}, industry={p1['phase1_industry']}")
        print(f"    Phase 2: model={s['business_model'] or '(empty)'}, industry={s['industry'] or '(empty)'}")
        
        model_fixed = s["business_model"] != p1["phase1_model"] or not s["business_model"]
        industry_fixed = s["industry"] != p1["phase1_industry"] or s["industry"] == p1["phase1_industry"]
        print(f"    Status: {'FIXED' if model_fixed else 'STILL WRONG'}")

# === L3 SIGNAL COVERAGE ===
print("\n" + "=" * 60)
print("L3 SIGNAL COVERAGE")
print("=" * 60)

l3_fields = ["retention_proxy", "onboarding_friction", "monetization_strength", 
             "competition_intensity", "funding_stage", "market_context"]

for field in l3_fields:
    filled = sum(1 for s in data["startups"] if s.get(field))
    pct = filled / total * 100 if total > 0 else 0
    print(f"  {field:<25}: {filled:>3}/{total} ({pct:>5.1f}%)")

# === FAILURE REASON QUALITY ===
print("\n" + "=" * 60)
print("FAILURE REASON EXTRACTION")
print("=" * 60)

with_failures = [s for s in data["startups"] if s.get("failure_reasons")]
print(f"  Entries with failure reasons: {len(with_failures)}/{total}")

for s in with_failures[:5]:
    print(f"\n  [{s['startup_name']}]")
    for r in s["failure_reasons"][:3]:
        print(f"    -> {r[:100]}")

# === DIFFERENTIATION CHECK ===
print("\n" + "=" * 60)
print("SaaS vs Fintech vs Marketplace DIFFERENTIATION")
print("=" * 60)

for model_type in ["SaaS", "Fintech", "Marketplace", "Devtools", "E-Commerce"]:
    matches = [s for s in data["startups"] if s.get("business_model") == model_type]
    if matches:
        print(f"\n  [{model_type}] ({len(matches)} entries)")
        for s in matches[:3]:
            print(f"    - {s['startup_name']}: {s['industry']}")
