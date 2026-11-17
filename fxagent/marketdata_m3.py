"""Audited offline historical BID/ASK tick-to-M15 import (no downloader).

Accepts normalized timestamp,bid,ask (UTC ISO8601), or dukascopy-node
CSV timestamp,askPrice,bidPrice[,askVolume,bidVolume] (UTC epoch ms).
Never invents missing bars or spreads. Ingestion is not evidence of tradability.
"""
import csv
import hashlib
import io
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal as D, InvalidOperation, localcontext
from pathlib import Path

from .data import Bar, Dataset, PIPS, STEP, validate, COLUMNS

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_NORMAL = ("timestamp", "bid", "ask")
_DUKA = ("timestamp", "askPrice", "bidPrice")


@dataclass(frozen=True)
class TickQualityPolicy:
    max_spread_pips: D = D("6")
    max_open_delay_seconds: int = 180
    max_close_delay_seconds: int = 180
    min_quotes_per_bar: int = 2

    def __post_init__(self):
        if not isinstance(self.max_spread_pips, D) or not self.max_spread_pips.is_finite() or self.max_spread_pips <= 0:
            raise ValueError("max_spread_pips must be a positive finite Decimal")
        if type(self.max_open_delay_seconds) is not int or not 0 <= self.max_open_delay_seconds < 900:
            raise ValueError("invalid max_open_delay_seconds")
        if type(self.max_close_delay_seconds) is not int or not 0 <= self.max_close_delay_seconds < 900:
            raise ValueError("invalid max_close_delay_seconds")
        if type(self.min_quotes_per_bar) is not int or self.min_quotes_per_bar < 2:
            raise ValueError("min_quotes_per_bar must be >= 2")


def _utc_tick(text, mode):
    if mode == "dukascopy-node-tick":
        if not text.isdecimal():
            raise ValueError("Dukascopy timestamp must be integer Unix milliseconds")
        when = _EPOCH + timedelta(milliseconds=int(text))
    else:
        when = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if when.tzinfo is None or when.utcoffset() != timedelta(0):
            raise ValueError("Normalized timestamps must have UTC timezone")
        when = when.astimezone(timezone.utc)
    return when


def _bar_start(t):
    return t.replace(minute=(t.minute // 15) * 15, second=0, microsecond=0)


def _make_bar(ticks, symbol):
    start = _bar_start(ticks[0][0])
    midpoint = [(b + a) / D(2) for _, b, a in ticks]
    max_spread = max((a - b) / PIPS[symbol] for _, b, a in ticks)
    return Bar(start, midpoint[0], max(midpoint), min(midpoint), midpoint[-1], max_spread)


def import_bidask_ticks(path, symbol, *, provider, provenance_url, permission_note,
                        policy=None, max_rows=2_000_000):
    """Return (raw Dataset, quality audit). No internet; caller provides market export.

    Quality findings are reported, *not silently repaired*. Call require_quality()
    before using the dataset in research. Tick CSV must be strictly chronological.
    """
    if symbol not in PIPS:
        raise ValueError("unsupported Forex pair")
    for value in (provider, provenance_url, permission_note):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("provider, provenance_url and permission_note required")
    if not isinstance(max_rows, int) or type(max_rows) is bool or max_rows < 1:
        raise ValueError("max_rows must be a positive integer")
    policy = TickQualityPolicy() if policy is None else policy
    if not isinstance(policy, TickQualityPolicy):
        raise ValueError("TickQualityPolicy required")
    raw = Path(path).read_bytes()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        rows = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
        fields = rows.fieldnames or []
        if fields == list(_NORMAL):
            mode, bid_col, ask_col = "normalized-utc", "bid", "ask"
        elif fields in (list(_DUKA), [*_DUKA, "askVolume", "bidVolume"]):
            mode, bid_col, ask_col = "dukascopy-node-tick", "bidPrice", "askPrice"
        else:
            raise ValueError("Expected timestamp,bid,ask OR timestamp,askPrice,bidPrice[,askVolume,bidVolume]")
        prev = None
        batch = []
        bars = []
        quality_flags = []
        spread_observations = []
        count = 0
        for line_no, row in enumerate(rows, 2):
            count += 1
            if count > max_rows:
                raise ValueError("Too many rows: use a smaller historical interval")
            if None in row or any(v is None or v.strip() == "" for v in row.values()):
                raise ValueError(f"row {line_no}: malformed quote")
            try:
                t = _utc_tick(row["timestamp"], mode)
                bid, ask = D(row[bid_col]), D(row[ask_col])
            except (ValueError, TypeError, InvalidOperation, OverflowError) as exc:
                raise ValueError(f"row {line_no}: invalid timestamp or quote: {exc}") from exc
            if not bid.is_finite() or not ask.is_finite() or bid <= 0 or ask < bid:
                raise ValueError(f"row {line_no}: require 0 < bid <= ask and finite quotes")
            if prev is not None and t <= prev:
                raise ValueError(f"row {line_no}: duplicate or non-increasing timestamp")
            prev = t
            start = _bar_start(t)
            if batch and start != _bar_start(batch[0][0]):
                bars.append(_make_bar(batch, symbol))
                _audit_bar(batch, bars[-1], symbol, policy, quality_flags)
                batch = []
            batch.append((t, bid, ask))
            spread_observations.append((ask - bid) / PIPS[symbol])
        if batch:
            bars.append(_make_bar(batch, symbol))
            _audit_bar(batch, bars[-1], symbol, policy, quality_flags)
    except UnicodeError as exc:
        raise ValueError("Tick CSV must be UTF-8") from exc
    if not bars:
        raise ValueError("No historical quotes available")
    validate(tuple(bars), symbol)
    missing = sum(int((b.time - a.time) / STEP) - 1 for a, b in zip(bars, bars[1:]))
    gap_examples = [
        {"previous": a.time.isoformat(), "next": b.time.isoformat(),
         "missing_m15_bars": int((b.time - a.time) / STEP) - 1}
        for a, b in zip(bars, bars[1:]) if b.time - a.time != STEP
    ]
    ordered = sorted(spread_observations)
    ninety_fifth = ordered[min(len(ordered) - 1, (len(ordered) * 95 + 99) // 100 - 1)]
    audit = {
        "schema": "fx-agent-m3-tick-audit-v1", "symbol": symbol,
        "provider": provider, "provenance_url": provenance_url,
        "permission_note": permission_note,
        "input_format": mode, "input_sha256": raw_sha256,
        "price_basis": "midpoint ticks", "spread_basis": "maximum observed full bid/ask spread per M15 bar",
        "quote_rows": count, "m15_bars": len(bars),
        "range_first": bars[0].time.isoformat(), "range_last": bars[-1].time.isoformat(),
        "max_spread_pips": str(ordered[-1]), "p95_spread_pips": str(ninety_fifth),
        "missing_m15_intervals_including_market_closures": missing,
        "gap_examples": gap_examples[:15], "gap_count": len(gap_examples),
        "flagged_bars": len(quality_flags), "flags": quality_flags[:25],
        "policy": {"max_spread_pips": str(policy.max_spread_pips),
                   "max_open_delay_seconds": policy.max_open_delay_seconds,
                   "max_close_delay_seconds": policy.max_close_delay_seconds,
                   "min_quotes_per_bar": policy.min_quotes_per_bar},
        "quality_pass": not quality_flags,
        "warnings": ["Missing intervals include normal market closures; review each unexpected gap.",
                     "Historical bid/ask snapshots do not prove executable fills or complete tick coverage.",
                     "Check your provider's license; do not redistribute source data without permission."]
    }
    dataset = Dataset(symbol, tuple(bars), "raw", f"{provider}; {provenance_url}; {permission_note}", raw_sha256)
    return dataset, audit


def _audit_bar(ticks, bar, symbol, policy, flags):
    reasons = []
    if len(ticks) < policy.min_quotes_per_bar:
        reasons.append("sparse_quotes")
    open_lag = (ticks[0][0] - bar.time).total_seconds()
    close_lag = (bar.time + STEP - ticks[-1][0]).total_seconds()
    if open_lag > policy.max_open_delay_seconds:
        reasons.append("late_open_quote")
    if close_lag > policy.max_close_delay_seconds:
        reasons.append("early_last_quote")
    if bar.spread_pips > policy.max_spread_pips:
        reasons.append("excessive_spread")
    if reasons:
        flags.append({"time": bar.time.isoformat(), "issues": reasons,
                      "quotes": len(ticks), "open_lag_seconds": open_lag,
                      "close_lag_seconds": close_lag,
                      "max_spread_pips": str(bar.spread_pips)})


def require_quality(audit):
    if not isinstance(audit, dict) or audit.get("schema") != "fx-agent-m3-tick-audit-v1":
        raise ValueError("Milestone 3 quality audit required")
    if not audit.get("quality_pass"):
        raise ValueError(f"Market data audit flagged {audit.get('flagged_bars')} bars; review before research")


def export_m15(dataset, path):
    """Write original Milestone 1 CSV schema without changing existing modules."""
    if dataset.role != "raw":
        raise ValueError("export raw datasets only")
    validate(dataset.bars, dataset.symbol)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for b in dataset.bars:
            writer.writerow([b.time.isoformat(), b.open, b.high, b.low, b.close, b.spread_pips])
    return Path(path)
