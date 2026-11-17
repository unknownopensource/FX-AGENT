"""Frozen-candidate rolling walk-forward using existing Milestone 2 simulator.

Search is restricted to training windows. Each validation window is used once
with the selected train-only configuration. Final chronological holdout is
excluded in its entirety and never passed to simulate_paper.
"""
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal as D

from .data import Dataset, fingerprint, validate, utc_time
from .paper import PaperConfig, simulate_paper
from .stats_m3 import descriptive_trade_stats


@dataclass(frozen=True)
class ResearchPlan:
    test_start: str
    train_bars: int
    validation_bars: int
    step_bars: int
    embargo_bars: int = 8
    min_train_trades: int = 3
    min_validation_trades: int = 30
    max_folds: int = 20

    def __post_init__(self):
        utc_time(self.test_start)
        for name in ("train_bars", "validation_bars", "step_bars", "embargo_bars",
                     "min_train_trades", "min_validation_trades", "max_folds"):
            v = getattr(self, name)
            if type(v) is not int or v < (0 if name == "embargo_bars" else 1):
                raise ValueError(f"{name} must be a nonnegative or positive integer")
        if self.train_bars < 64 or self.validation_bars < 64:
            raise ValueError("train and validation must each contain at least 64 M15 bars")
        if self.step_bars < self.validation_bars:
            raise ValueError("overlapping validation windows are prohibited")
        if self.embargo_bars < 8:
            raise ValueError("embargo must be >= 8 M15 bars to purge max hold time")
        if self.max_folds > 100:
            raise ValueError("max_folds may not exceed 100")


@dataclass(frozen=True)
class Candidate:
    name: str
    config: PaperConfig

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 64:
            raise ValueError("candidate name required")
        if not isinstance(self.config, PaperConfig):
            raise ValueError("PaperConfig required")


@dataclass(frozen=True)
class CostScenario:
    name: str
    spread_factor: D
    slippage_factor: D
    commission_factor: D

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("scenario name required")
        for name in ("spread_factor", "slippage_factor", "commission_factor"):
            v = getattr(self, name)
            if not isinstance(v, D) or not v.is_finite() or v < 1:
                raise ValueError(f"{name} must be a finite Decimal >= 1")


DEFAULT_COSTS = (
    CostScenario("base", D(1), D(1), D(1)),
    CostScenario("spread2_slip2", D(2), D(2), D(1)),
    CostScenario("all_costs3", D(3), D(3), D(3)),
)


def isolate_pre_holdout(data, test_start):
    """Returns pretest research data only; no test bars in returned object."""
    if data.role != "raw":
        raise ValueError("Only raw dataset may be divided once into research/holdout")
    validate(data.bars, data.symbol)
    boundary = datetime.fromisoformat(test_start.replace("Z", "+00:00"))
    boundary = utc_time(boundary.isoformat())
    research_bars = tuple(bar for bar in data.bars if bar.time < boundary)
    holdout_count = len(data.bars) - len(research_bars)
    if not research_bars or holdout_count < 1:
        raise ValueError("Require nonempty preholdout and nonempty untouched holdout")
    research = replace(data, role="train", bars=research_bars)
    return research, {
        "test_start": boundary.isoformat(), "holdout_bars_excluded": holdout_count,
        "research_bars": len(research_bars), "research_sha256": fingerprint(research),
        "holdout_not_evaluated": True,
        "note": "Only a workflow guard, not a security sandbox. Store holdout separately.",
    }


def _fold_slices(research, plan):
    if research.role != "train":
        raise ValueError("Fold generation accepts preholdout research data only")
    validate(research.bars, research.symbol)
    if research.bars[-1].time >= utc_time(plan.test_start):
        raise ValueError("Research data contains holdout timestamps")
    if len(research.bars) < plan.train_bars + plan.embargo_bars + plan.validation_bars:
        raise ValueError("Insufficient preholdout M15 data for first fold")
    start = 0
    fold = 0
    while fold < plan.max_folds:
        train_end = start + plan.train_bars
        val_start = train_end + plan.embargo_bars
        val_end = val_start + plan.validation_bars
        if val_end > len(research.bars):
            break
        train = replace(research, role="train", bars=research.bars[start:train_end])
        validation = replace(research, role="validation", bars=research.bars[val_start:val_end])
        if train.bars[-1].time >= validation.bars[0].time:
            raise AssertionError("train/validation leakage")
        yield train, validation
        start += plan.step_bars
        fold += 1


def _stress_dataset(validation, scenario):
    return replace(validation, bars=tuple(
        replace(bar, spread_pips=bar.spread_pips * scenario.spread_factor)
        for bar in validation.bars
    ))


def run_walk_forward(research, plan, candidates, scenarios=DEFAULT_COSTS):
    """Train selection -> one untouched validation run per fold -> stress tests.

    The caller must pass only the preholdout subset from isolate_pre_holdout.
    Strictly rejects raw and test roles, duplicate candidate/scenario names,
    weaker-than-baseline cost scenarios, and crossover past test_start.
    """
    if not isinstance(plan, ResearchPlan):
        raise ValueError("ResearchPlan required")
    candidates = tuple(candidates)
    scenarios = tuple(scenarios)
    if not candidates or not scenarios:
        raise ValueError("Candidates and scenarios must be nonempty")
    if any(not isinstance(c, Candidate) for c in candidates):
        raise ValueError("Candidate required")
    if len({c.name for c in candidates}) != len(candidates):
        raise ValueError("Candidate names must be unique")
    if any(not isinstance(s, CostScenario) for s in scenarios):
        raise ValueError("CostScenario required")
    if len({s.name for s in scenarios}) != len(scenarios):
        raise ValueError("Scenario names must be unique")
    if scenarios[0] != CostScenario("base", D(1), D(1), D(1)):
        raise ValueError("First scenario must be explicit base costs")
    if len(candidates) > 25:
        raise ValueError("Too many search candidates; use a prespecified small grid")
    if research.role != "train":
        raise ValueError("Test/raw/validation datasets cannot enter research runner")
    if plan.embargo_bars < max(c.config.max_hold_bars for c in candidates):
        raise ValueError("embargo must cover the longest candidate holding horizon")
    if research.bars[-1].time >= utc_time(plan.test_start):
        raise ValueError("Holdout bars detected")
    folds = []
    all_val_trades = []
    for index, (train, validation) in enumerate(_fold_slices(research, plan), start=1):
        leaderboard = []
        for candidate in candidates:
            result = simulate_paper(train, candidate.config)
            leaderboard.append((candidate, result))
        eligible = [(c, r) for c, r in leaderboard if r["number_of_trades"] >= plan.min_train_trades]
        fold = {
            "fold": index,
            "train_start": train.bars[0].time.isoformat(),
            "train_end_exclusive": (train.bars[-1].time + timedelta(minutes=15)).isoformat(),
            "validation_start": validation.bars[0].time.isoformat(),
            "validation_end_exclusive": (validation.bars[-1].time + timedelta(minutes=15)).isoformat(),
            "embargo_bars_requested": plan.embargo_bars,
            "train_sha256": fingerprint(train), "validation_sha256": fingerprint(validation),
            "train_candidates": [{"name": c.name, "train_net_pnl": r["net_pnl"],
                                   "train_trades": r["number_of_trades"]} for c, r in leaderboard],
            "selected": None, "validation": None, "cost_stress": [],
        }
        if eligible:
            # Chronological selection uses TRAIN results only. Stable tie-breaker
            # is the order of prespecified candidates, not validation performance.
            chosen, training = max(eligible, key=lambda cr: D(cr[1]["net_pnl"]))
            fold["selected"] = chosen.name
            fold["train_selected_net_pnl"] = training["net_pnl"]
            validation_result = simulate_paper(validation, chosen.config)
            fold["validation"] = {
                "net_pnl": validation_result["net_pnl"],
                "number_of_trades": validation_result["number_of_trades"],
                "max_drawdown_fraction": validation_result["max_drawdown_fraction"],
                "trades": validation_result["trades"],
            }
            all_val_trades.extend(validation_result["trades"])
            for scenario in scenarios:
                if scenario.name == "base":
                    result = validation_result
                else:
                    config = replace(chosen.config,
                        slippage_pips=chosen.config.slippage_pips * scenario.slippage_factor,
                        commission_per_unit=chosen.config.commission_per_unit * scenario.commission_factor)
                    result = simulate_paper(_stress_dataset(validation, scenario), config)
                fold["cost_stress"].append({
                    "scenario": scenario.name,
                    "spread_factor": str(scenario.spread_factor),
                    "slippage_factor": str(scenario.slippage_factor),
                    "commission_factor": str(scenario.commission_factor),
                    "net_pnl": result["net_pnl"],
                    "trades": result["number_of_trades"],
                })
        folds.append(fold)
    if not folds:
        raise ValueError("No research folds; check minimum history, plan and test boundary")
    stress_aggregate = {
        s.name: {"net_pnl_quote_currency": str(sum(
            (D(f["cost_stress"][i]["net_pnl"]) for f in folds if f["selected"] is not None),
            D(0))),
            "folds_evaluated": sum(f["selected"] is not None for f in folds)}
        for i, s in enumerate(scenarios)
    }
    return {
        "schema": "fx-agent-walkforward-m3-v1", "symbol": research.symbol,
        "account_currency": research.symbol[3:],
        "source_sha256": research.source_sha256,
        "research_sha256": fingerprint(research),
        "test_start": plan.test_start,
        "test_partition_backtested": False,
        "config": {k: getattr(plan, k) for k in (
            "test_start", "train_bars", "validation_bars", "step_bars",
            "embargo_bars", "min_train_trades", "min_validation_trades", "max_folds")},
        "candidate_names": [c.name for c in candidates],
        "candidate_trials_per_fold": len(candidates),
        "folds": folds,
        "selected_folds": sum(f["selected"] is not None for f in folds),
        "validation_metrics": descriptive_trade_stats(all_val_trades,
                                                       min_required=plan.min_validation_trades),
        "aggregate_cost_stress": stress_aggregate,
        "methodology": [
            "Rolling, non-overlapping chronological validation windows; train-only candidate selection.",
            "At least 8 embargo M15 bars separate train and validation; no cross-fold positions.",
            "No history carryover into validation: cold-start H1/H4 context; early bars have no signals.",
            "Each fold resets cash; aggregate PnL is a sum of independent simulations in quote currency.",
            "Final holdout untouched and unavailable to this evaluator; not a security boundary.",
            "Realized PnL and CI are descriptive; repeated selection, regime drift and dependence remain risks.",
            "Cost scenarios re-simulate executions, not just subtract a fixed fee; trade sets may differ.",
        ]
    }
