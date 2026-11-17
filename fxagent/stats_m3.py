"""Descriptive out-of-sample diagnostics, not proofs of edge or significance."""
import math
import random
from decimal import Decimal as D


def wilson_interval(wins, observations, z=1.959963984540054):
    if type(wins) is not int or type(observations) is not int or not 0 <= wins <= observations:
        raise ValueError("invalid win count")
    if not observations:
        return None
    p = wins / observations
    denom = 1 + z * z / observations
    middle = (p + z * z / (2 * observations)) / denom
    radius = z * math.sqrt((p * (1-p) + z*z/(4*observations))/observations) / denom
    return [round(max(0, middle - radius), 6), round(min(1, middle + radius), 6)]


def moving_block_mean_ci(values, *, samples=400, block_size=None, seed=73):
    """95% resampled mean-trade PnL CI; conditional and descriptive only."""
    n = len(values)
    if n < 2:
        return None
    if type(samples) is not int or not 100 <= samples <= 10000:
        raise ValueError("samples must be 100..10000")
    block_size = max(1, int(math.sqrt(n))) if block_size is None else block_size
    if type(block_size) is not int or not 1 <= block_size <= n:
        raise ValueError("invalid block_size")
    numbers = [float(D(str(x))) for x in values]
    if any(not math.isfinite(v) for v in numbers):
        raise ValueError("nonfinite trade pnl")
    rng = random.Random(seed)
    averages = []
    for _ in range(samples):
        draw = []
        while len(draw) < n:
            start = rng.randrange(n)
            draw.extend(numbers[(start+j) % n] for j in range(block_size))
        averages.append(sum(draw[:n]) / n)
    averages.sort()
    return [round(averages[int(.025 * (samples-1))], 8),
            round(averages[int(.975 * (samples-1))], 8)]


def descriptive_trade_stats(trades, *, min_required=30):
    if type(min_required) is not int or min_required < 2:
        raise ValueError("min_required must be >= 2")
    pnls = [D(t["net_pnl"]) for t in trades]
    if any(not p.is_finite() for p in pnls):
        raise ValueError("nonfinite net_pnl")
    total = sum(pnls, D(0))
    win_count = sum(p > 0 for p in pnls)
    gains = sum((p for p in pnls if p > 0), D(0))
    losses = -sum((p for p in pnls if p < 0), D(0))
    return {
        "trades": len(trades), "net_pnl_quote_currency": str(total),
        "mean_trade_pnl": str(total / len(pnls)) if pnls else None,
        "wins": win_count,
        "win_rate": round(win_count/len(pnls), 6) if pnls else None,
        "wilson_win_rate_95_interval": wilson_interval(win_count, len(pnls)),
        "moving_block_bootstrap_mean_pnl_95_interval": moving_block_mean_ci(pnls),
        "profit_factor": str(gains / losses) if losses > 0 else None,
        "minimum_validation_trades_required": min_required,
        "evidence_gate": "descriptive_sample_only" if len(pnls) >= min_required else "insufficient_trades",
        "interpretation": (
            "Exploratory metrics only; NOT a p-value, confidence in profitability, "
            "or proof of trading edge. Selection and cross-fold dependence can "
            "invalidate naive inference; no family-wise multiple-testing correction."
        ),
    }
