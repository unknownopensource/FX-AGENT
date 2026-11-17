from decimal import Decimal, Context, localcontext

from .data import PIPS, STEP, fingerprint, validate
from .strategy import wants_long

D = Decimal


def simulate(data, *, initial_cash, units, slippage_pips, commission_per_unit):
    """Offline, long/flat, one-bar simulation.

    Cash and commission are denominated in the symbol's quote currency.
    Units are units of the base currency.
    """
    if data.role not in ("train", "validation"):
        raise ValueError("Backtesting raw or untouched test data is prohibited.")

    with localcontext(Context(prec=28)):
        validate(data.bars, data.symbol)
        for name, value in (
            ("initial_cash", initial_cash),
            ("slippage_pips", slippage_pips),
            ("commission_per_unit", commission_per_unit),
        ):
            if not isinstance(value, D) or not value.is_finite() or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative Decimal.")
        if initial_cash <= 0:
            raise ValueError("Initial cash must be positive.")
        if type(units) is not int or units <= 0:
            raise ValueError("Units must be a positive integer.")

        # Fixed simulator rule; not exposed as a strategy parameter.
        maximum_capital_fraction = D("0.20")
        cash = initial_cash
        peak = cash
        maximum_drawdown = D("0")
        trades = []
        bars = data.bars
        equity = [{
            "time": bars[0].time.isoformat(),
            "equity": str(cash),
        }]

        for index, bar in enumerate(bars):
            consecutive = index >= 2 and all(
                bars[j].time - bars[j - 1].time == STEP
                for j in (index - 1, index)
            )

            if consecutive and wants_long(
                tuple(b.close for b in bars[index - 2:index])
            ):
                half_spread = bar.spread_pips * PIPS[data.symbol] / 2
                slippage = slippage_pips * PIPS[data.symbol]
                entry = bar.open + half_spread + slippage
                exit_price = bar.close - half_spread - slippage
                fee_per_side = D(units) * commission_per_unit

                if bar.low <= half_spread + slippage:
                    raise ValueError("Costs imply a nonpositive executable price.")

                capital_used = entry * units + fee_per_side
                if capital_used > cash * maximum_capital_fraction:
                    raise ValueError(
                        "Position exceeds the fixed 20% capital-deployment cap."
                    )

                # Buy base currency, then sell it at this bar's close.
                cash -= capital_used
                cash += exit_price * units - fee_per_side

                gross = (bar.close - bar.open) * units
                spread_cost = 2 * half_spread * units
                slippage_cost = 2 * slippage * units
                commission_cost = 2 * fee_per_side
                net = gross - spread_cost - slippage_cost - commission_cost

                trades.append({
                    "entry_time": bar.time.isoformat(),
                    "exit_time": (bar.time + STEP).isoformat(),
                    "units": units,
                    "entry_price": str(entry),
                    "exit_price": str(exit_price),
                    "gross_pnl": str(gross),
                    "spread_cost": str(spread_cost),
                    "slippage_cost": str(slippage_cost),
                    "commission_cost": str(commission_cost),
                    "net_pnl": str(net),
                })

            peak = max(peak, cash)
            maximum_drawdown = max(maximum_drawdown, (peak - cash) / peak)
            equity.append({
                "time": (bar.time + STEP).isoformat(),
                "equity": str(cash),
            })

        return {
            "strategy": "two_completed_closes_long_one_bar_v1",
            "symbol": data.symbol,
            "account_currency": data.symbol[3:],
            "partition": data.role,
            "source": data.source,
            "source_sha256": data.source_sha256,
            "partition_sha256": fingerprint(data),
            "configuration": {
                "initial_cash": str(initial_cash),
                "units": units,
                "slippage_pips_per_side": str(slippage_pips),
                "commission_per_base_unit_per_side": str(commission_per_unit),
                "maximum_capital_fraction": str(maximum_capital_fraction),
            },
            "assumptions": [
                "Midpoint M15 OHLC; timestamps identify bar openings.",
                "Reported full spread applies at both entry and exit.",
                "Constant adverse slippage per side; full fills.",
                "Long/flat; every position closes within its entry bar.",
                "No financing, margin, market impact, or currency conversion.",
            ],
            "bars": len(bars),
            "gaps": sum(
                current.time - previous.time != STEP
                for previous, current in zip(bars, bars[1:])
            ),
            "number_of_trades": len(trades),
            "final_cash": str(cash),
            "net_pnl": str(cash - initial_cash),
            "return_fraction": str(cash / initial_cash - 1),
            "bar_close_max_drawdown_fraction": str(maximum_drawdown),
            "open_positions_at_end": 0,
            "trades": trades,
            "equity": equity,
        }
