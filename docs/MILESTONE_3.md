# FX-AGENT Milestone 3 — offline data audit and research only

## Scope and safety

- Additive files: **Milestones 1 and 2 are untouched**. No broker, order execution,
  margin, or paid AI API. Python 3.10+, standard library only.
- Historical **bid/ask ticks** are required for spread-aware imports. Bid-only
  M1 or bid-only M15 does not let us infer actual spread. Never relabel a
  bid-only candle as a midpoint candle.
- Real data are **not bundled or fetched**. Obtain your own CSV via an approved
  provider export, verify its accuracy and permitted use, then upload it
  privately into Colab. Never commit ticks, `data/`, or `reports/` to public GitHub.
- Every event is historical. No live prices, automatic trades, or return claims.
- Quote-currency accounting is unchanged from M2 (JPY for USDJPY, etc.).
  Do not add results across different pair denominations.

## Accepted tick CSV formats

**Format A: normalized** (UTC ISO8601, strictly increasing, bid/ask quotes):

```csv
timestamp,bid,ask
2024-01-08T00:00:10Z,1.09995,1.10005
2024-01-08T00:14:00Z,1.10005,1.10015
```

**Format B: `dukascopy-node` tick CSV** (UTC Unix milliseconds, includes both sides):

```csv
timestamp,askPrice,bidPrice,askVolume,bidVolume
1704672010000,1.10005,1.09995,1,1
1704672840000,1.10015,1.10005,1,1
```

Also accepts the first 3 columns of Format B. **Not** ordinary candle/M1 CSV.
The open and close are first/last observed tick midpoints within a UTC M15
bucket, not guaranteed executable boundary quotes; the highest full observed
spread of that bucket is applied for the whole bar conservatively. Quality
audit checks sparse buckets, late initial quotes, early final quotes, high
spreads, and gaps (including normal weekends); it never fills or repairs
missing data. `quality_pass` is a gate for user research. You must separately
inspect gaps, outliers, feed licensing, timestamp convention and vendor data
quality. `source_sha256` records exact original CSV bytes.

### Obtaining historical data for free

One option is the Dukascopy Historical Data export page:
https://www.dukascopy.com/swiss/english/marketwatch/historical/

If using the third-party open-source `dukascopy-node` downloader, **its tick**
CSV format `timestamp,askPrice,bidPrice,askVolume,bidVolume` is supported.
Read vendor usage terms. The `dukascopy-node` tool is not required to run this
project and is not bundled. Start with small research intervals. Data-feed
cost and licensing differ from software cost. GitHub/Colab availability can
change independently from the source provider.

## Workflow — zero paid AI API

1. In Colab, open the Milestone 3 installer notebook and upload the M3 ZIP.
2. Clone the repository, run prior tests, and install new files in a new
   temporary folder. Verify older file SHA-256 hashes are unmodified.
3. Run full test suite and the clearly marked **synthetic** demonstration.
4. Use the optional real-data cell only when you have an actual historical
   tick CSV. Provide a provider name, source URL, license/use note, a fixed
   `test_start` boundary, and enough history for non-overlapping folds.
5. Publish **code only** to a `milestone-3-review-*` GitHub branch. Review PR
   and merge to `main` only after independently checking output. No source
   market data, results, secrets, or notebook runtime outputs are published.

### Running real research from terminal/Colab

Run from the project root, replacing the sample dates/path with your data:

```bash
python -m fxagent.run_m3 \
  --input /content/your_ticks.csv --symbol EURUSD \
  --provider "Dukascopy historical export" \
  --source-url "https://www.dukascopy.com/swiss/english/marketwatch/historical/" \
  --permission-note "Reviewed provider terms; personal research use only" \
  --test-start 2024-05-01T00:00:00Z \
  --train-bars 960 --validation-bars 480 --step-bars 480
```

The default quality policy requires >=2 quotes per M15 bucket, first tick <=
3 minutes from opening, last tick <=3 minutes from closing, and maximum tick
spread <=6 pips (may be inappropriate for some pairs/sessions; review before
changing). If any bar fails, only the audit JSON is saved and research stops.

## Walk-forward protocol

- `isolate_pre_holdout` returns a **train-labeled research dataset only**. It
  excludes the final test period from evaluation. Save or access test data
  separately for a truly isolated final evaluation: a Python label is a guard,
  not access control. Final holdout is **never tested** in Milestone 3.
- `ResearchPlan` uses **rolling** training windows with a purged M15 embargo
  at least as long as the largest candidate holding horizon and at least 8 bars.
  Validation windows cannot overlap. All folds remain strictly pre-holdout.
- Select a predetermined candidate on **training PnL** with a required minimum
  trade count; evaluate it once on validation. Fold equity is reset and H1/H4
  context cold-starts; no boundary state or future candles cross partitions.
- Cost-stress scenarios resimulate at baseline, 2x spread & slippage, and
  3x spread/slippage/commission. Changed costs can change trade sets.
- Reports include hashes, fold boundaries, train-only candidates and frozen
  selections, validation trade ledgers, drawdowns, cost scenarios, Wilson win
  intervals and a seeded moving-block bootstrap of mean trade PnL. These
  confidence intervals are **descriptive only**; they do not correct data
  snooping, repeated strategy search, regime selection, or all trade dependence.
- If fewer than 30 validation trades were collected, report `insufficient_trades`.
  Even more trades are **not** evidence of an independently verified edge.

## Known limitations

Observed top-of-book ticks, M15 OHLC exits, full fills, fixed slippage,
no margin/financing/partial fills/market impact. Stop-first behavior when
stop and target can both occur within a candle. Unobserved intrabar paths
and real market liquidity are not reconstructed. No annualized Sharpe or
p-values presented, since their assumptions are unjustified here. No USD
conversion across mixed pairs. No auto-retraining, AI agent, or live broker.

## Commands

```bash
python -m unittest discover -s tests -v
python -m fxagent.demo_m3
```

No external packages or API tokens needed for tests/demo. Synthetic demo files
are placed only under ignored `data/` and `reports/`.
