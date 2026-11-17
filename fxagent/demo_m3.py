"""Completely synthetic offline demonstration; NOT historical performance."""
import csv
import json
import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from pathlib import Path

from .data import STEP
from .marketdata_m3 import import_bidask_ticks, require_quality, export_m15
from .walkforward_m3 import (Candidate, ResearchPlan, isolate_pre_holdout,
                             run_walk_forward)
from .paper import PaperConfig


def synthetic_quotes(path, count=1400):
    start = datetime(2024, 1, 8, tzinfo=timezone.utc)
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["timestamp", "bid", "ask"])
        for i in range(count):
            t = start + i * STEP
            # Designed test series; NO real prices or econometric significance.
            level = D("1.10000") + D(i % 256 - 128) * D("0.000015")
            level += D(str(round(math.sin(i / 13) * .0002, 7)))
            for off, movement in ((10, D("0")), (400, D("0.0001")), (840, D("-0.00005"))):
                mid = level + movement
                spread = D("0.00012")
                writer.writerow([(t + timedelta(seconds=off)).isoformat(),
                                 mid-spread/2, mid+spread/2])
    return start


def main():
    root = Path("data")
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "synthetic_bidask_m3.csv"
    start = synthetic_quotes(csv_path)
    data, audit = import_bidask_ticks(csv_path, "EURUSD", provider="SYNTHETIC DEMO",
                                     provenance_url="https://example.invalid/synthetic",
                                     permission_note="Generated offline; not market data")
    require_quality(audit)
    export_m15(data, root / "synthetic_m15_m3.csv")
    research, sealed = isolate_pre_holdout(data, (start + 1250 * STEP).isoformat())
    candidates = [
        Candidate(f"fixed_{stop}_{target}", PaperConfig(
            initial_cash=D("10000"), units=1000, stop_pips=D(stop),
            target_pips=D(target), slippage_pips=D("0.5"),
            commission_per_unit=D("0.0001"), max_hold_bars=6))
        for stop, target in ((8, 18), (10, 25), (14, 28))
    ]
    plan = ResearchPlan(test_start=(start + 1250 * STEP).isoformat(),
                        train_bars=300, validation_bars=192, step_bars=192,
                        embargo_bars=8, min_train_trades=1)
    result = run_walk_forward(research, plan, candidates)
    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out / "synthetic_m3_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (out / "synthetic_m3_walkforward.json").write_text(json.dumps({
        "holdout_manifest": sealed, "research_result": result}, indent=2), encoding="utf-8")
    print("SYNTHETIC DATA ONLY; NOT A PROFITABILITY CLAIM")
    print("M15 bars:", audit["m15_bars"], "| Audit passed:", audit["quality_pass"])
    print("Research folds:", len(result["folds"]), "| Selected:", result["selected_folds"])
    print("Validation:", result["validation_metrics"]["evidence_gate"],
          "| PnL:", result["validation_metrics"]["net_pnl_quote_currency"],
          result["account_currency"])
    print("Untouched holdout bars:", sealed["holdout_bars_excluded"])
    print("Reports:", out.resolve())


if __name__ == "__main__":
    main()
