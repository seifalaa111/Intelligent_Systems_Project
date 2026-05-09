"""
MIDAN Data Pipeline -- Operations Drift Monitor
Tracks pipeline execution variances and alerts against heavy systemic CONDITIONAL loops or broken absolute tendencies.
"""

import json
from pathlib import Path
from collections import defaultdict
from rich.console import Console

console = Console()

def monitor_drift(json_path: Path):
    if not json_path.exists():
        console.print(f"[red]No export path found at {json_path}[/red]")
        return
        
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    entries = data.get("startups", []) if isinstance(data, dict) else data
    if not entries:
         console.print("[yellow]Empty telemetry array.[/yellow]")
         return
         
    dist = defaultdict(int)
    confidence_sum = 0.0
    
    for e in entries:
        decision = e.get("decision_analysis", {}).get("decision", "UNKNOWN")
        conf = e.get("decision_analysis", {}).get("decision_confidence", 0.0)
        dist[decision] += 1
        confidence_sum += conf
        
    total = len(entries)
    
    console.print(f"\n[bold cyan]── DECISION DRIFT TELEMETRY ({total} startups) ──[/bold cyan]")
    
    conditional_pct = 0.0
    for k, v in dist.items():
        pct = (v / total) * 100
        console.print(f" - {k}: {pct:.1f}% ({v}/{total})")
        if k == "CONDITIONAL":
            conditional_pct = pct

    avg_conf = (confidence_sum / total) * 100
    console.print(f"\n[bold]Mean Global Confidence:[/bold] {avg_conf:.1f}%")
    
    # Alerts
    if conditional_pct > 70.0:
         console.print("\n[red][ALERT] CONDITIONAL drift exceeded > 70%. Evaluate pipeline metrics for data starvation or over-indexing penalties.[/red]")
    elif conditional_pct < 5.0 and len(dist.keys()) == 1:
         console.print("\n[red][ALERT] Systemic skew detected. Single unverified decision structure dominating pipeline completely.[/red]")
    else:
         console.print("\n[green][OK] Pipeline distribution securely balanced.[/green]")

if __name__ == "__main__":
    monitor_drift(Path("data/structured/startup_intelligence.json"))
