"""
MIDAN Data Pipeline -- System Stability Tester
Validates strict deterministic parsing behavior guaranteeing outputs hold exactly across iterations.
Target 10 structurally diverse startups (SaaS, FinTech, Hardware/Media / Failures).
"""

import os
import json
from collections import defaultdict
from pathlib import Path
from rich.console import Console

console = Console()

def load_json(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def run_stability():
    console.print("\n[bold cyan]>> PHASE 3 STABILITY TESTING[/bold cyan]")
    
    # Run the pipeline programmatically 3 times targeting the 10 samples.
    # To keep this test standalone and not pollute actual datasets, we will run the main CLI via os.system.
    targets = ["Slack", "Notion", "Linear", "Figma", "Vercel", "Stripe", "Airtable", "Quibi", "Atrium", "Fast"]
    
    # We will write a temporary target list since we cannot supply arbitrary lists through CLI easily, 
    # but we can just use the provided startup_targets.json if it contains these, or we can just 
    # run `--sources website failory` for all 20 targets 3 times, which tests more data.
    
    OUTPUT_PATHS = [
        "data/structured/run_1.json",
        "data/structured/run_2.json",
        "data/structured/run_3.json",
        "data/structured/run_4.json",
        "data/structured/run_5.json"
    ]
    
    for i, p in enumerate(OUTPUT_PATHS):
        console.print(f"[yellow]Running structural loop {i+1}...[/yellow]")
        cmd = f"python main.py --sources website failory --limit 10 --output {p} > NUL 2>&1"
        os.system(cmd)
        
    outputs = [load_json(p) for p in OUTPUT_PATHS]
    
    if any(not o for o in outputs):
        console.print("[red]Critical Error: Runs failed to populate JSON.[/red]")
        return
        
    # Analyze Drift
    entries_list = [o["startups"] for o in outputs]
    
    drifts_detected = 0
    drift_log = []
    
    base_entries = entries_list[0]
    
    for entry in base_entries:
        name = entry.get("startup_name", "").lower()
        
        # Verify across all other sets
        for i in range(1, len(entries_list)):
            comparison_set = entries_list[i]
            # Find matching startup
            match = next((e for e in comparison_set if e.get("startup_name", "").lower() == name), None)
            
            if not match:
                drifts_detected += 1
                drift_log.append(f"MIA: {name} completely missing in run {i+1}")
                continue
                
            for k, v in entry.items():
                if k in ["extracted_on", "source_url"]:
                    continue
                match_val = match.get(k)
                if v != match_val:
                    drifts_detected += 1
                    drift_log.append(f"DRIFT at {name} -> Key [{k}]: '{v}' != '{match_val}'")
                    
    if drifts_detected == 0:
        console.print("\n[green][OK] STABILITY TEST PASSED. 0 DEVIATIONS ACROSS 5 RUNS.[/green]")
    else:
        console.print(f"\n[red]FAILED: {drifts_detected} structural drifts detected.[/red]")
        for log in drift_log[:10]:
            safe_log = log.encode("ascii", "ignore").decode("ascii")
            console.print(f" - {safe_log}")
            
    # Cleanup
    for p in OUTPUT_PATHS:
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    run_stability()


def test_run_stability_smoke(tmp_path, monkeypatch):
    workdir = tmp_path / "workspace"
    structured = workdir / "data" / "structured"
    structured.mkdir(parents=True, exist_ok=True)

    def fake_system(cmd):
        output_path = cmd.split("--output ", 1)[1].split(" > ", 1)[0].strip()
        target = workdir / output_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "startups": [
                {
                    "startup_name": "Slack",
                    "primary_industry": "Enterprise Software",
                    "business_model": "SaaS",
                    "source_url": "https://example.com/slack",
                }
            ]
        }
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return 0

    monkeypatch.chdir(workdir)
    monkeypatch.setattr(os, "system", fake_system)

    run_stability()
    remaining = list(structured.glob("run_*.json"))
    assert remaining == []
