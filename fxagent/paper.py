"""Deterministic historical PAPER fills. No orders, network, broker or AI hooks.

Midpoint OHLC + per-bar constant full spread. Conservative same-bar stop/target
policy. Account currency is the quote currency; not a multi-pair USD portfolio.
"""
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal as D, Context, localcontext

from .data import PIPS, STEP, fingerprint, validate
from .timeframes import aggregate_completed, CompletedContext

# Unoverrideable conservative engineering ceilings, NOT investment advice.
_MAX_NOTIONAL = D("0.20")
_MAX_ENTRY_RISK = D("0.01")
_DAILY_LOSS_LIMIT = D("0.02")


@dataclass(frozen=True)
class PaperConfig:
    initial_cash: D
    units: int                        # positive units of base currency
    stop_pips: D
    target_pips: D
    slippage_pips: D
    commission_per_unit: D            # quote currency per base unit, per side
    max_hold_bars: int = 8


def _check_config(config):
    if not isinstance(config, PaperConfig):
        raise ValueError("PaperConfig required")
    if type(config.units) is not int or config.units < 1:
        raise ValueError("units must be a positive integer")
    if type(config.max_hold_bars) is not int or not 1 <= config.max_hold_bars <= 96:
        raise ValueError("max_hold_bars must be between 1 and 96")
    for name in ("initial_cash", "stop_pips", "target_pips", "slippage_pips", "commission_per_unit"):
        value = getattr(config, name)
        if not isinstance(value, D) or not value.is_finite() or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative Decimal")
    if config.initial_cash == 0 or config.stop_pips == 0 or config.target_pips == 0:
        raise ValueError("initial_cash, stop_pips, target_pips must be positive")


def _price(mid: D, spread_pips: D, pip: D, slip: D, side: str) -> D:
    """Adverse executable price, including spread and slippage."""
    half = spread_pips * pip / 2
    return mid + half + slip if side == "buy" else mid - half - slip


def _exit_decision(position, bar, pip, slip):
    """Stop/target are executable levels, checking current bar extremities.

    On a gap through a stop, fill at worse open; same-bar stop wins if both hit.
    This is still an OHLC-only approximation, not tick-level execution.
    """
    is_long = position["side"] == "long"
    exit_side = "sell" if is_long else "buy"
    opening = _price(bar.open, bar.spread_pips, pip, slip, exit_side)
    low_exec = _price(bar.low, bar.spread_pips, pip, slip, exit_side)
    high_exec = _price(bar.high, bar.spread_pips, pip, slip, exit_side)
    stop, target = position["stop"], position["target"]
    if is_long:
        if opening <= stop:
            return (opening, "stop_gap")
        if low_exec <= stop:
            return (stop, "stop")
        if high_exec >= target:
            return (target, "target")
    else:
        if opening >= stop:
            return (opening, "stop_gap")
        if high_exec >= stop:
            return (stop, "stop")
        if low_exec <= target:
            return (target, "target")
    return None


def simulate_paper(data, config: PaperConfig):
    """Offline M15+H1/H4 intraday long/short research simulation.

    Signal uses prior two finished M15 closes and H1/H4 closes available as of
    the current M15 open. One open position at a time, one side per symbol.
    Skips entries exceeding fixed 20% notional / 1% estimated risk ceilings.
    When daily realized drawdown exceeds 2%, stops *new entries* until next UTC
    day. Open positions are managed by stops and timeouts. The daily limiter is
    NOT a guaranteed portfolio loss ceiling (gaps can exceed stops).
    """
    if data.role not in ("train", "validation"):
        raise ValueError("Only train/validation allowed; raw/test prohibited")
    with localcontext(Context(prec=28)):
        validate(data.bars, data.symbol)
        _check_config(config)
        bars, pip = data.bars, PIPS[data.symbol]
        one_hour = CompletedContext(aggregate_completed(bars, data.symbol, 1), 1)
        four_hour = CompletedContext(aggregate_completed(bars, data.symbol, 4), 4)
        cash, peak = config.initial_cash, config.initial_cash
        slip = config.slippage_pips * pip
        units = D(config.units)
        fee = units * config.commission_per_unit
        trades, equity = [], []
        position = None
        daily_cash_open = cash
        last_day = None
        max_drawdown = D(0)
        skipped = {"risk": 0, "daily_loss": 0, "stale_context": 0}

        def liquidation(mid, spread, side):
            exit_side = "sell" if side == "long" else "buy"
            return _price(mid, spread, pip, slip, exit_side)

        def close_position(price, reason, event_time):
            nonlocal cash, position
            if price <= 0:
                raise ValueError("Nonpositive executable exit price")
            sign = D(1) if position["side"] == "long" else D(-1)
            gross = sign * (price - position["entry"]) * units
            net = gross - 2 * fee
            cash += net
            trades.append({
                "side": position["side"],
                "entry_time": position["entry_time"].isoformat(),
                "exit_time": event_time.isoformat(),
                "entry_price": str(position["entry"]),
                "exit_price": str(price),
                "stop_price": str(position["stop"]),
                "target_price": str(position["target"]),
                "units": config.units,
                "gross_pnl_after_spread_slippage": str(gross),
                "commission_total": str(2 * fee),
                "net_pnl": str(net),
                "exit_reason": reason,
            })
            position = None

        for index, bar in enumerate(bars):
            if bar.time.date() != last_day:
                daily_cash_open = cash
                last_day = bar.time.date()
            contiguous = index > 0 and bar.time - bars[index - 1].time == STEP
            if position is not None and not contiguous:
                close_position(liquidation(bar.open, bar.spread_pips, position["side"]),
                               "gap_liquidation", bar.time)

            if position is not None:
                decision = _exit_decision(position, bar, pip, slip)
                position["bars_held"] += 1
                if decision is not None:
                    close_position(decision[0], decision[1], bar.time + STEP)
                elif position["bars_held"] >= config.max_hold_bars:
                    close_position(liquidation(bar.close, bar.spread_pips, position["side"]),
                                   "timeout", bar.time + STEP)

            # If a position was just closed, do not re-enter on same M15 bar.
            was_closed = bool(trades and trades[-1]["exit_time"] == (bar.time + STEP).isoformat())
            if position is None and not was_closed and index >= 2 and contiguous:
                history_ok = bars[index - 1].time - bars[index - 2].time == STEP
                h1 = one_hour.direction_at(bar.time)
                h4 = four_hour.direction_at(bar.time)
                if not history_ok or not h1 or not h4:
                    skipped["stale_context"] += 1
                else:
                    change = (bars[index - 1].close > bars[index - 2].close) - (
                        bars[index - 1].close < bars[index - 2].close)
                    side = "long" if change == h1 == h4 == 1 else (
                        "short" if change == h1 == h4 == -1 else None)
                    if side is not None:
                        if cash <= daily_cash_open * (D(1) - _DAILY_LOSS_LIMIT):
                            skipped["daily_loss"] += 1
                        else:
                            entry_side = "buy" if side == "long" else "sell"
                            entry = _price(bar.open, bar.spread_pips, pip, slip, entry_side)
                            if entry <= 0:
                                raise ValueError("Nonpositive executable entry price")
                            stop_distance = config.stop_pips * pip
                            target_distance = config.target_pips * pip
                            stop = entry - stop_distance if side == "long" else entry + stop_distance
                            target = entry + target_distance if side == "long" else entry - target_distance
                            if stop <= 0 or target <= 0:
                                raise ValueError("Invalid stop or target price")
                            opposite_side = "sell" if side == "long" else "buy"
                            immediate_exit = _price(bar.open, bar.spread_pips, pip, slip, opposite_side)
                            liquidation_loss = abs(entry - immediate_exit) * units
                            estimated_risk = max(stop_distance * units, liquidation_loss) + 2 * fee
                            # No leverage, fixed quote-denominated capital accounting.
                            if (entry * units + fee > cash * _MAX_NOTIONAL or
                                    estimated_risk > cash * _MAX_ENTRY_RISK):
                                skipped["risk"] += 1
                            else:
                                position = {
                                    "side": side, "entry": entry, "stop": stop,
                                    "target": target, "entry_time": bar.time,
                                    "bars_held": 0,
                                }
                                # A trade entered at this bar's open can hit stop/target
                                # within this bar. Conservative stop-first assumption.
                                decision = _exit_decision(position, bar, pip, slip)
                                position["bars_held"] += 1
                                if decision is not None:
                                    close_position(decision[0], decision[1], bar.time + STEP)
                                elif config.max_hold_bars == 1:
                                    close_position(liquidation(bar.close, bar.spread_pips, side),
                                                   "timeout", bar.time + STEP)

            if position is not None and index == len(bars) - 1:
                close_position(liquidation(bar.close, bar.spread_pips, position["side"]),
                               "end_of_data", bar.time + STEP)

            # Mark to executable liquidation value if still open; no phantom P&L.
            equity_now = cash
            if position is not None:
                mark = liquidation(bar.close, bar.spread_pips, position["side"])
                sign = D(1) if position["side"] == "long" else D(-1)
                equity_now += sign * (mark - position["entry"]) * units - 2 * fee
            peak = max(peak, equity_now)
            max_drawdown = max(max_drawdown, (peak - equity_now) / peak)
            equity.append({"time": (bar.time + STEP).isoformat(),
                           "equity": str(equity_now), "cash": str(cash),
                           "open_side": None if position is None else position["side"]})

        return {
            "engine": "offline_paper_m2_v1",
            "symbol": data.symbol,
            "partition": data.role,
            "account_currency": data.symbol[3:],
            "source": data.source,
            "source_sha256": data.source_sha256,
            "partition_sha256": fingerprint(data),
            "config": {name: str(getattr(config, name)) for name in (
                "initial_cash", "units", "stop_pips", "target_pips",
                "slippage_pips", "commission_per_unit", "max_hold_bars")},
            "risk_caps": {"max_notional_fraction": str(_MAX_NOTIONAL),
                          "max_estimated_entry_risk_fraction": str(_MAX_ENTRY_RISK),
                          "daily_realized_loss_entry_halt_fraction": str(_DAILY_LOSS_LIMIT)},
            "trades": trades,
            "equity": equity,
            "number_of_trades": len(trades),
            "skipped_signals": skipped,
            "final_cash": str(cash),
            "net_pnl": str(cash - config.initial_cash),
            "max_drawdown_fraction": str(max_drawdown),
            "open_positions_at_end": 0,
            "assumptions": [
                "Synthetic/offline midpoint OHLC; full spread constant within each M15 bar.",
                "H1 and H4 signals only from complete and fresh closed candles.",
                "Stop hit wins if both stop and target are inside one bar.",
                "Gap-through stops execute at worse observed open; not a price guarantee.",
                "Full fills, fixed adverse slippage, commission on entry and exit.",
                "No margin, financing, liquidity, partial fills, market impact, or FX conversion.",
                "Daily loss gate blocks new entries only, not guaranteed maximum loss.",
                "Prices/cash in quote currency; no portfolio cross-currency aggregation.",
            ],
        }
