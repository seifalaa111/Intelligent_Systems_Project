"""
MIDAN Data Pipeline -- Phase 3 Signal Validation Engine
Compares pipeline output against ground truth dataset to calculate accuracy metrics.
Logs failures structurally to error_log.json to prevent hallucinations.
"""

import json
import os
from collections import defaultdict
from rich.console import Console

console = Console()

GROUND_TRUTH_PATH = os.path.join("data", "validation", "ground_truth.json")
PIPELINE_OUTPUT_PATH = os.path.join("data", "structured", "startup_intelligence.json")
ERROR_LOG_PATH = os.path.join("data", "validation", "error_log.json")


def load_json(path):
    if not os.path.exists(path):
        console.print(f"[red]Error: Missing file {path}[/red]")
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            console.print(f"[red]Error: Invalid JSON in {path}[/red]")
            return []


def validate():
    ground_truth = load_json(GROUND_TRUTH_PATH)
    pipeline_data = load_json(PIPELINE_OUTPUT_PATH)
    if not ground_truth or not pipeline_data:
        return

    # Extract the entries depending on the wrapper
    # pipeline_data is usually a dict: {"total_entries": ..., "entries": [...]}
    if isinstance(pipeline_data, dict) and "startups" in pipeline_data:
        pipeline_entries = pipeline_data["startups"]
    else:
        pipeline_entries = pipeline_data

    # Create mapping of startup name to pipeline entry
    pipeline_map = {entry.get("startup_name", "").lower(): entry for entry in pipeline_entries}

    metrics = defaultdict(lambda: {"correct": 0, "total": 0})
    global_correct = 0
    global_total = 0
    errors = []

    console.print("\n[bold cyan]>> PHASE 3 SIGNAL VALIDATION ENGINE[/bold cyan]")
    
    for gt_entry in ground_truth:
        startup_name = gt_entry["startup_name"].lower()
        if startup_name not in pipeline_map:
            console.print(f"[yellow]Skipping {gt_entry['startup_name']}: Not found in pipeline output[/yellow]")
            continue

        predicted_entry = pipeline_map[startup_name]

        for expected in gt_entry.get("signals", []):
            signal_key = expected["signal"]
            actual_val = str(expected["value"]).lower().strip()
            pred_val = str(predicted_entry.get(signal_key, "")).lower().strip()
            
            # Special handling for lists like failure_reasons
            if isinstance(predicted_entry.get(signal_key), list):
                pred_val = " | ".join([str(v).lower() for v in predicted_entry.get(signal_key)])

            global_total += 1
            metrics[signal_key]["total"] += 1

            # Determine Match
            match = False
            if actual_val in pred_val:
                match = True
            
            if match:
                global_correct += 1
                metrics[signal_key]["correct"] += 1
            else:
                if pred_val == "":
                    etype = "missing_evidence"
                elif signal_key in ["primary_industry", "business_model"]:
                    etype = "classification_error"
                else:
                    etype = "signal_misinterpretation"

                errors.append({
                    "error_type": etype,
                    "startup": gt_entry["startup_name"],
                    "signal": signal_key,
                    "expected": actual_val,
                    "predicted": pred_val,
                    "justification": expected.get("justification", "")
                })

    # Group taxonomy for output JSON
    taxonomy_log = defaultdict(list)
    for err in errors:
        taxonomy_log[err["error_type"]].append({k: v for k, v in err.items() if k != "error_type"})

    # Print Table
    console.print("\n[bold]Validation Metrics by Signal:[/bold]")
    for signal, data in metrics.items():
        if data["total"] > 0:
            acc = (data["correct"] / data["total"]) * 100
            color = "green" if acc >= 80 else "red"
            console.print(f"  - {signal}: [{color}]{acc:.1f}%[/{color}] ({data['correct']}/{data['total']})")

    if global_total > 0:
        global_acc = (global_correct / global_total) * 100
        color = "green" if global_acc >= 80 else "red"
        console.print(f"\n[bold]Global Accuracy:[/bold] [{color}]{global_acc:.1f}%[/{color}] ({global_correct}/{global_total})")
    
    # Save errors
    with open(ERROR_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(taxonomy_log, f, indent=2)
    
    if errors:
        console.print(f"[yellow]Logged {len(errors)} mismatches into taxonomic structure across {ERROR_LOG_PATH}[/yellow]")
    else:
        console.print("[green]Perfect alignment reached![/green]")

if __name__ == "__main__":
    validate()
