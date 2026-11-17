"""Manual real historical CSV intake; no network or hidden downloads.

Example:
 python -m fxagent.run_m3 --input data/my_ticks.csv --symbol EURUSD \
   --provider Dukascopy --source-url https://www.dukascopy.com/ \
   --permission-note 'Personal research; terms reviewed' \
   --test-start 2024-05-01T00:00:00Z
"""
import argparse
import json
from datetime import datetime, timezone
from decimal import Decimal as D
from pathlib import Path
from .marketdata_m3 import import_bidask_ticks, require_quality, export_m15, TickQualityPolicy
from .walkforward_m3 import Candidate, ResearchPlan, isolate_pre_holdout, run_walk_forward
from .paper import PaperConfig


def main():
    parser = argparse.ArgumentParser(description="Manual offline FX-AGENT M3 historical research")
    for arg in ("input", "symbol", "provider", "source-url", "permission-note", "test-start"):
        parser.add_argument("--" + arg, required=True)
    parser.add_argument("--max-spread-pips", type=str, default="6")
    parser.add_argument("--train-bars", type=int, default=960)
    parser.add_argument("--validation-bars", type=int, default=480)
    parser.add_argument("--step-bars", type=int, default=480)
    parser.add_argument("--embargo-bars", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=2000000)
    args = parser.parse_args()
    out = Path("reports")
    out.mkdir(exist_ok=True)
    policy = TickQualityPolicy(max_spread_pips=D(args.max_spread_pips))
    data, audit = import_bidask_ticks(args.input, args.symbol,
        provider=args.provider, provenance_url=args.source_url,
        permission_note=args.permission_note, policy=policy, max_rows=args.max_rows)
    audit_path = out / "m3_real_data_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("Audit saved:", audit_path, "| flagged bars:", audit["flagged_bars"])
    require_quality(audit)
    m15_path = Path("data") / "historical_converted_M15.csv"
    m15_path.parent.mkdir(exist_ok=True)
    export_m15(data, m15_path)
    research, sealed = isolate_pre_holdout(data, args.test_start)
    candidates = [Candidate(f"fixed_{s}_{t}", PaperConfig(
        initial_cash=D("10000"), units=1000, stop_pips=D(s), target_pips=D(t),
        slippage_pips=D("0.5"), commission_per_unit=D("0.0001"), max_hold_bars=6))
        for s, t in ((8, 18), (10, 25), (14, 28))]
    plan = ResearchPlan(test_start=args.test_start, train_bars=args.train_bars,
                        validation_bars=args.validation_bars, step_bars=args.step_bars,
                        embargo_bars=args.embargo_bars)
    result = run_walk_forward(research, plan, candidates)
    path = out / "m3_real_research.json"
    path.write_text(json.dumps({"holdout_manifest": sealed,
                                 "research_result": result}, indent=2), encoding="utf-8")
    print("Research complete. Final holdout untouched.")
    print("Validation evidence gate:", result["validation_metrics"]["evidence_gate"])
    print("Report:", path)


if __name__ == "__main__":
    main()
