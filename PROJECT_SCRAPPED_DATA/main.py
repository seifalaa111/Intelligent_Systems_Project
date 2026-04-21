"""
MIDAN Data Pipeline -- Main Orchestrator (Phase 2)
Website + Failory + YC + Reddit + Product Hunt collection and extraction.
Optional local LLM (Ollama) supplement for ambiguous narratives.

Usage:
    python main.py                              # Run full pipeline with default targets
    python main.py --sources website yc         # Run specific collectors only
    python main.py --sources website yc reddit  # Include Reddit collector
    python main.py --targets custom.json        # Use custom target list
    python main.py --yc-only                    # Only collect from YC directory
    python main.py --failory-only               # Only collect from Failory
    python main.py --limit 5                    # Limit website targets to 5
    python main.py --no-llm                     # Disable Ollama LLM
"""

import json
import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime, timezone

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from itertools import cycle
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from config.settings import (
    BASE_DIR, STRUCTURED_DIR, TARGETS_DIR, OUTPUT_FILE, LOG_FILE,
)
from collectors.website_collector import WebsiteCollector
from collectors.failory_collector import FailoryCollector
from collectors.yc_collector import YCCollector
from processors.source_classifier import classify_source, get_available_extractors
from processors.schema_validator import validate_batch, clean_entry
from processors.conflict_resolver import ConflictResolver
from processors.pattern_analyzer import PatternAnalyzer
from processors.decision_engine import DecisionEngine
from utils.logger import setup_logger, log_stage, log_success, log_error

console = Console()
logger = setup_logger("pipeline", LOG_FILE)

# Phase 2 sources -- imported conditionally
PHASE2_SOURCES_AVAILABLE = {}

try:
    from collectors.reddit_collector import RedditCollector
    PHASE2_SOURCES_AVAILABLE["reddit"] = True
except ImportError:
    PHASE2_SOURCES_AVAILABLE["reddit"] = False

try:
    from collectors.producthunt_collector import ProductHuntCollector
    PHASE2_SOURCES_AVAILABLE["producthunt"] = True
except ImportError:
    PHASE2_SOURCES_AVAILABLE["producthunt"] = False


ALL_SOURCES = ["website", "failory", "yc", "reddit", "producthunt"]


def load_targets(targets_file: str | None = None) -> list[dict]:
    """Load startup targets from JSON file."""
    if targets_file:
        filepath = Path(targets_file)
    else:
        filepath = TARGETS_DIR / "startup_targets.json"

    if not filepath.exists():
        logger.warning(f"Targets file not found: {filepath}")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    targets = data.get("targets", [])
    logger.info(f"Loaded {len(targets)} targets from {filepath.name}")
    return targets


def run_collectors(targets: list[dict], sources: list[str] | None = None,
                   limit: int | None = None) -> list[dict]:
    """Run selected collectors and return all raw data."""
    all_raw = []
    sources = sources or ["website", "failory", "yc"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:

        # -- WEBSITE COLLECTOR --
        if "website" in sources and targets:
            website_targets = targets[:limit] if limit else targets
            task = progress.add_task(
                f"[cyan]Collecting websites ({len(website_targets)} targets)...",
                total=len(website_targets),
            )
            collector = WebsiteCollector()
            raw = collector.collect(website_targets)
            all_raw.extend(raw)
            progress.update(task, completed=len(website_targets))
            _print_collector_report(collector)

        # -- FAILORY COLLECTOR --
        if "failory" in sources:
            task = progress.add_task("[magenta]Collecting Failory case studies...", total=1)
            collector = FailoryCollector()
            raw = collector.collect()
            all_raw.extend(raw)
            progress.update(task, completed=1)
            _print_collector_report(collector)

        # -- YC COLLECTOR --
        if "yc" in sources:
            task = progress.add_task("[green]Collecting YC directory...", total=1)
            collector = YCCollector()
            if targets:
                raw = collector.collect(targets)
            else:
                raw = collector.collect()
            all_raw.extend(raw)
            progress.update(task, completed=1)
            _print_collector_report(collector)

        # -- REDDIT COLLECTOR (Phase 2) --
        if "reddit" in sources:
            if PHASE2_SOURCES_AVAILABLE.get("reddit"):
                task = progress.add_task("[yellow]Collecting Reddit discussions...", total=1)
                collector = RedditCollector()
                raw = collector.collect(targets)
                all_raw.extend(raw)
                progress.update(task, completed=1)
                _print_collector_report(collector)
            else:
                console.print("  [yellow](!)[/yellow] Reddit collector skipped (PRAW not installed)")

        # -- PRODUCT HUNT COLLECTOR (Phase 2) --
        if "producthunt" in sources:
            if PHASE2_SOURCES_AVAILABLE.get("producthunt"):
                ph_targets = targets[:limit] if limit else targets
                task = progress.add_task(
                    f"[blue]Collecting Product Hunt ({len(ph_targets)} targets)...",
                    total=len(ph_targets),
                )
                collector = ProductHuntCollector()
                raw = collector.collect(ph_targets)
                all_raw.extend(raw)
                progress.update(task, completed=len(ph_targets))
                _print_collector_report(collector)
            else:
                console.print("  [yellow](!)[/yellow] Product Hunt collector skipped (import error)")

    return all_raw


def run_extraction(raw_entries: list[dict], use_llm: bool = False) -> list[dict]:
    """Run context-aware extraction on all raw entries."""
    extractors = get_available_extractors()
    structured = []

    console.print(f"\n[bold cyan]>> EXTRACTION[/bold cyan] -- Processing {len(raw_entries)} raw entries\n")

    # Initialize Ollama if requested
    ollama = None
    if use_llm:
        try:
            from utils.ollama_client import get_ollama_client
            ollama = get_ollama_client()
            if ollama.available:
                console.print("  [green][OK][/green] Ollama LLM active -- supplementing rule-based extraction")
            else:
                console.print("  [yellow](!)[/yellow] Ollama unavailable -- using rule-based only")
                ollama = None
        except Exception as e:
            console.print(f"  [yellow](!)[/yellow] Ollama init failed: {e}")
            ollama = None

    for raw_entry in raw_entries:
        try:
            # Classify and route
            extractor_type = classify_source(raw_entry)
            extractor = extractors.get(extractor_type)

            if not extractor:
                logger.warning(f"No extractor for type '{extractor_type}', skipping")
                continue

            # Extract structured fields
            entry = extractor.extract(raw_entry)

            # LLM supplement for ambiguous insight entries
            if ollama and extractor_type == "insight":
                entry = _llm_supplement(ollama, entry, raw_entry)

            # Clean the entry (includes normalization)
            entry = clean_entry(entry)

            structured.append(entry)

        except Exception as e:
            log_error(logger, f"Extraction failed for {raw_entry.get('startup_name', '?')}: {e}")

    console.print(f"  [green][OK][/green] Extracted {len(structured)} entries")
    return structured


def _llm_supplement(ollama, entry: dict, raw_entry: dict) -> dict:
    """
    Supplement rule-based extraction with LLM for ambiguous entries.
    ONLY fills empty fields -- never overwrites rule-based results.
    """
    raw_content = raw_entry.get("raw_content", "")
    startup_name = entry.get("startup_name", "")

    # Supplement 1: (REMOVED) Business model inference explicitly forbidden in Phase 2 Final.
    
    # Supplement 2: Extract failure reasoning from Failory narratives
    source_url = raw_entry.get("source_url", "")
    if "failory" in source_url.lower() and not entry.get("failure_reasons"):
        llm_result = ollama.extract_failure_reasoning(raw_content, startup_name)
        if llm_result.get("failure_reasons"):
            entry["failure_reasons"] = llm_result["failure_reasons"]
        if llm_result.get("funding_stage") and not entry.get("funding_stage"):
            entry["funding_stage"] = llm_result["funding_stage"]
        if llm_result.get("market_context") and not entry.get("market_context"):
            entry["market_context"] = llm_result["market_context"]

    return entry


def run_validation(entries: list[dict]) -> list[dict]:
    """Validate all entries and return only valid ones."""
    console.print(f"\n[bold cyan]>> VALIDATION[/bold cyan] -- Checking {len(entries)} entries\n")

    valid, invalid, stats = validate_batch(entries)

    # Print validation stats
    table = Table(title="Validation Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Total entries", str(stats["total"]))
    table.add_row("Valid", str(stats["valid"]))
    table.add_row("Invalid", str(stats["invalid"]))
    table.add_row("Total issues", str(stats["total_issues"]))
    console.print(table)

    if invalid:
        console.print(f"\n  [yellow](!)[/yellow] Invalid entries:")
        for inv in invalid[:5]:
            name = inv["entry"].get("startup_name", "unknown")
            issues = "; ".join(inv["issues"][:3])
            console.print(f"    [red][FAIL][/red] {name}: {issues}")

    return valid


def deduplicate(entries: list[dict]) -> list[dict]:
    """
    Merge entries for the same startup from different sources.
    Uses ConflictResolver to compute confidence scores and determine hard/soft conflicts.
    """
    console.print(f"\n[bold cyan]>> CONFLICT RESOLUTION & MERGE[/bold cyan]\n")

    # Group entries by startup name
    grouped = defaultdict(list)
    for entry in entries:
        name = entry["startup_name"].lower().strip()
        grouped[name].append(entry)

    resolver = ConflictResolver()
    results = []

    for name, group in grouped.items():
        resolved_entry = resolver.resolve(group)
        results.append(resolved_entry)

    console.print(f"  [green][OK][/green] {len(entries)} entries merged into {len(results)} unique startups")
    return results


def save_output(entries: list[dict], output_file: Path | None = None):
    """Save structured entries to JSON."""
    output_path = output_file or OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": "2.0.0-phase2",
            "total_entries": len(entries),
            "sources_used": list(set(e.get("source_type", "") for e in entries)),
            "schema_fields": [
                "startup_name", "primary_industry", "secondary_industry", "business_model", "target_user",
                "target_segment", "value_proposition",
                "pain_points", "adoption_barriers", "user_complaints",
                "success_drivers", "failure_reasons",
                "differentiation", "switching_cost", "competition_density",
                "signal_evidence", "confidence_score",
                "retention_proxy", "onboarding_friction",
                "monetization_strength", "competition_intensity",
                "funding_stage", "market_context",
                "system_patterns", "pattern_confidence_score",
                "pattern_success_rates", "pattern_failure_rates",
                "decision_analysis",
                "source_type", "source_url",
            ],
            "extracted_on": datetime.now().isoformat(),
        },
        "startups": entries,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    console.print(f"\n  [green][OK][/green] Saved {len(entries)} entries to {output_path}")
    return output_path


def print_sample(entries: list[dict], n: int = 3):
    """Print sample entries for quick inspection."""
    console.print(f"\n[bold cyan]>> SAMPLE OUTPUT[/bold cyan] -- {min(n, len(entries))} entries\n")

    for entry in entries[:n]:
        panel_content = ""
        for key, value in entry.items():
            if isinstance(value, list):
                if value:
                    panel_content += f"[cyan]{key}:[/cyan]\n"
                    for item in value[:3]:
                        panel_content += f"  - {item[:100]}\n"
                else:
                    panel_content += f"[cyan]{key}:[/cyan] [dim](empty)[/dim]\n"
            else:
                display_val = str(value)[:120] if value else "[dim](empty)[/dim]"
                panel_content += f"[cyan]{key}:[/cyan] {display_val}\n"

        console.print(Panel(
            panel_content,
            title=f"[bold]{entry.get('startup_name', 'Unknown')}[/bold]",
            border_style="green",
        ))


def main():
    """Main pipeline orchestrator."""
    parser = argparse.ArgumentParser(description="MIDAN Data Acquisition Pipeline (Phase 2)")
    parser.add_argument("--sources", nargs="+", default=["website", "failory", "yc"],
                       choices=ALL_SOURCES,
                       help="Which collectors to run")
    parser.add_argument("--targets", type=str, default=None,
                       help="Path to custom targets JSON file")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of website/PH targets to process")
    parser.add_argument("--yc-only", action="store_true",
                       help="Only run YC collector")
    parser.add_argument("--failory-only", action="store_true",
                       help="Only run Failory collector")
    parser.add_argument("--output", type=str, default=None,
                       help="Custom output file path")
    parser.add_argument("--sample", type=int, default=3,
                       help="Number of sample entries to print")
    parser.add_argument("--use-llm", action="store_true",
                       help="Enable Ollama LLM to supplement extraction")
    parser.add_argument("--no-llm", action="store_true",
                       help="Explicitly disable LLM (default)")
    args = parser.parse_args()

    # Handle shortcut flags
    if args.yc_only:
        args.sources = ["yc"]
    elif args.failory_only:
        args.sources = ["failory"]

    use_llm = args.use_llm and not args.no_llm

    # -- BANNER --
    console.print(Panel(
        "[bold white]MIDAN - Startup Intelligence Data Pipeline[/bold white]\n"
        "[dim]Phase 2: Website + Failory + YC + Reddit + Product Hunt[/dim]\n"
        f"[dim]Sources: {', '.join(args.sources)}[/dim]\n"
        f"[dim]LLM: {'Enabled' if use_llm else 'Disabled'}[/dim]",
        border_style="bright_blue",
    ))

    start_time = time.time()

    # -- STEP 1: LOAD TARGETS --
    log_stage(logger, "STEP 1", "Loading targets")
    targets = load_targets(args.targets)

    # -- STEP 2: COLLECT RAW DATA --
    log_stage(logger, "STEP 2", "Running collectors")
    raw_data = run_collectors(targets, args.sources, args.limit)

    if not raw_data:
        console.print("\n[bold red]FAIL: No data collected. Check logs for errors.[/bold red]")
        sys.exit(1)

    console.print(f"\n  [green][OK][/green] Collected {len(raw_data)} raw entries total")

    # -- STEP 3: EXTRACT STRUCTURED DATA --
    log_stage(logger, "STEP 3", "Running extraction")
    structured = run_extraction(raw_data, use_llm=use_llm)

    # -- STEP 4: VALIDATE --
    log_stage(logger, "STEP 4", "Running validation")
    valid = run_validation(structured)

    # -- STEP 5: DEDUPLICATE --
    log_stage(logger, "STEP 5", "Deduplicating")
    final = deduplicate(valid)

    # -- STEP 6: SAVE OUTPUT --
    log_stage(logger, "STEP 6", "Saving output")
    # 9. Pattern Analysis
    console.print("\n[bold cyan]── ENFORCING BEHAVIORAL PATTERNS ──[/bold cyan]")
    analyzer = PatternAnalyzer()
    pattern_results = analyzer.analyze_patterns(final)

    # 10. Decision Engine
    console.print("\n[bold cyan]── EVALUATING STARTUP VIABILITY ──[/bold cyan]")
    decider = DecisionEngine()
    final_results = decider.evaluate_payload(pattern_results)

    # 11. Save Output
    output_path = save_output(final_results, Path(args.output) if args.output else None)

    # -- STEP 7: SAMPLE --
    print_sample(final, args.sample)

    # -- SUMMARY --
    elapsed = time.time() - start_time

    # Count L3 signal coverage
    l3_fields = ["retention_proxy", "onboarding_friction",
                 "monetization_strength", "competition_intensity"]
    l3_filled = sum(1 for e in final for f in l3_fields if e.get(f))
    l3_total = len(final) * len(l3_fields)
    l3_pct = (l3_filled / l3_total * 100) if l3_total > 0 else 0

    console.print(Panel(
        f"[bold green]Pipeline Complete[/bold green]\n\n"
        f"  Entries collected:  {len(raw_data)}\n"
        f"  Entries extracted:  {len(structured)}\n"
        f"  Entries validated:  {len(valid)}\n"
        f"  Unique startups:   {len(final)}\n"
        f"  L3 signal coverage: {l3_pct:.0f}% ({l3_filled}/{l3_total})\n"
        f"  Time elapsed:      {elapsed:.1f}s\n"
        f"  Output:            {output_path}",
        title="[bold]Pipeline Summary[/bold]",
        border_style="bright_green",
    ))


def _print_collector_report(collector):
    """Print a brief collector report."""
    report = collector.report()
    status = "[green][OK][/green]" if report["collected"] > 0 else "[red][FAIL][/red]"
    console.print(f"  {status} {report['source_type']}: "
                  f"{report['collected']} collected, {report['errors']} errors")


if __name__ == "__main__":
    main()
