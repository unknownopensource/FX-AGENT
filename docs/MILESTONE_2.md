# FX-AGENT Milestone 2 — Offline M15 / H1 / H4 Paper Research

**Status:** Engineering demonstration only. Neither AI nor any trading/broker integration is included. No real-money trading. No external Python runtime dependencies. Never claim profitability based on synthetic data.

## What was added (no Milestone 1 source changes)

- `fxagent/timeframes.py`: UTC-aligned, completed-only H1/H4 candles generated from complete M15 source windows. Missing source bars invalidate that higher-timeframe candle; stale contexts and incomplete windows do not produce trend signals.
- `fxagent/paper.py`: deterministic, **one-symbol at a time** and one position at a time, long/short, midpoint bar+spread/slippage simulator; stop/target conservative tie breaker; gap-through stop at worse observed bar open; hold timeout; final forced closure; cash/equity and transaction ledger. Signals require two completed M15 closes plus two completed, consecutive H1 and H4 candles with matching directional momentum.
- `fxagent/demo_m2.py`: creates **synthetic** EURUSD data and simulates only the `train` partition. It does not download real history.
- `tests/test_milestone2.py`: additional tests of aggregation, no-lookahead, stop/target semantics, quote currency, costs, config, sizing and report.
- `FX_AGENT_Milestone2_Installer.ipynb`: Chromebook/Colab installer and opt-in GitHub review branch publication.

### Guarantees and non-guarantees

- All positions are simulated from historical M15 price bars. There are no endpoints or broker interfaces, credentials, autonomous loops, network orders, or AI calls in the simulator.
- `simulate_paper` accepts only `train` or `validation` dataset roles. An untouched `test` (and `raw`) role is rejected. This is a workflow guard, **not a security boundary**; anyone editing Python can bypass it.
- Only fresh, fully closed H1/H4 candles may influence the bar-opening decision. An entry may only use previous completed M15 bars. A context gap blocks new entries until two fresh consecutive higher-timeframe candles are present.
- Full spread is treated as constant within each M15 bar. Both sides incur adverse slippage and commission. Since only OHLC data are known, if a stop and target are touched in the same bar the stop wins (conservative choice). Stop gaps can create losses larger than the configured stop distance.
- Hard-coded engineering entry ceilings: estimated risk <=1% of available cash, notional <=20%, and *new-entry only* halt after daily realized loss >=2% of daily opening cash (UTC day). These do **not** guarantee maximum loss. A large gap can exceed them; no leverage/margin modeling.
- Base units and cash are in different units: e.g. EURUSD results are in USD, USDJPY in JPY, USDCHF in CHF. Never sum returns or quote-currency P&L across symbols without conversion and a true portfolio ledger.
- Equity is a mark-to-liquidation estimate at bar close, including prospective liquidation cost and round-trip commission on an open position. There is no bid/ask tick data, execution latency, financing, market impact, order-book depth, partial fills, rollover, interest, market hours or liquidity constraints.
- No AI optimizer, no real historical data acquisition, no forward walk, no validated strategy, no calibrated position sizing.
- Source file unchanged: `fxagent/data.py`, `fxagent/backtest.py`, `fxagent/strategy.py`, `fxagent/reporting.py`, `fxagent/demo.py`, `fxagent/__init__.py`, `tests/test_foundation.py`.

## Run locally or in Colab

```bash
python -m unittest discover -s tests -v
python -m fxagent.demo_m2
```

Run from the repository root in Python 3.10+. The synthetic demo writes a CSV in `data/` and a report under `reports/` (both should be ignored by Git).

## Review-based GitHub installation

The installer notebook clones the public `unknownopensource/FX-AGENT` repository to a new temporary Colab directory, verifies the Milestone 1 tests, adds only **new Milestone 2 paths** locally, runs both demos and all tests, and then optionally creates a separate `milestone-2-review-*` branch using a repository-scoped GitHub token held in Colab Secrets. It **does not directly modify `main`**. Review and merge the pull request yourself.

Use Colab Secrets key `GITHUB_TOKEN`; GitHub fine-grained token needs `Contents: Read and write` on only this repository. Never store the token or generated reports in public GitHub. If `main` has moved since cloning, the installer aborts publication to avoid applying the patch onto an unexpected revision.
