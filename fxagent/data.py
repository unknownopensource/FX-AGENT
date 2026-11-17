import csv
import hashlib
import io
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType

D = Decimal
STEP = timedelta(minutes=15)
PIPS = MappingProxyType({
    "EURUSD": D("0.0001"),
    "GBPUSD": D("0.0001"),
    "USDJPY": D("0.01"),
    "AUDUSD": D("0.0001"),
    "USDCHF": D("0.0001"),
})
COLUMNS = ("timestamp", "open", "high", "low", "close", "spread_pips")


@dataclass(frozen=True)
class Bar:
    time: datetime                 # UTC opening time of the M15 bar
    open: Decimal                  # Midpoint OHLC
    high: Decimal
    low: Decimal
    close: Decimal
    spread_pips: Decimal           # Full spread, assumed constant within bar


@dataclass(frozen=True)
class Dataset:
    symbol: str
    bars: tuple[Bar, ...]
    role: str                      # raw, train, validation, or test
    source: str
    source_sha256: str


def utc_time(text):
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.utcoffset() != timedelta(0):
        raise ValueError("Timestamps must include a UTC timezone.")
    if value.minute % 15 or value.second or value.microsecond:
        raise ValueError("Timestamps must align to M15 boundaries.")
    return value.astimezone(timezone.utc)


def validate(bars, symbol):
    if symbol not in PIPS:
        raise ValueError(f"Unsupported symbol: {symbol}")
    if not bars:
        raise ValueError("Dataset is empty.")

    previous = None
    for index, bar in enumerate(bars, start=1):
        utc_time(bar.time.isoformat())
        values = (bar.open, bar.high, bar.low, bar.close, bar.spread_pips)
        if any(not isinstance(v, D) or not v.is_finite() for v in values):
            raise ValueError(f"Bar {index}: values must be finite Decimals.")
        if min(bar.open, bar.high, bar.low, bar.close) <= 0:
            raise ValueError(f"Bar {index}: prices must be positive.")
        if not bar.low <= min(bar.open, bar.close) <= max(bar.open, bar.close) <= bar.high:
            raise ValueError(f"Bar {index}: inconsistent OHLC prices.")
        if bar.spread_pips < 0:
            raise ValueError(f"Bar {index}: negative spread.")
        if bar.low <= bar.spread_pips * PIPS[symbol] / 2:
            raise ValueError(f"Bar {index}: spread implies a nonpositive bid.")
        if previous is not None and bar.time <= previous:
            raise ValueError("Timestamps must be strictly increasing and unique.")
        previous = bar.time
    # Gaps are preserved. Missing bars are never fabricated or forward-filled.


def load_csv(path, symbol, *, source, price_basis):
    if price_basis != "mid":
        raise ValueError("This version requires midpoint OHLC, not bid-only OHLC.")
    if not source.strip():
        raise ValueError("Provide a source description.")

    raw = Path(path).read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if reader.fieldnames != list(COLUMNS):
        raise ValueError(f"Required CSV columns, in order: {COLUMNS}")

    bars = []
    for line, row in enumerate(reader, start=2):
        if set(row) != set(COLUMNS) or any(v is None for v in row.values()):
            raise ValueError(f"Malformed CSV row {line}.")
        try:
            bars.append(Bar(
                utc_time(row["timestamp"]),
                *(D(row[name]) for name in COLUMNS[1:])
            ))
        except (ValueError, InvalidOperation, TypeError) as exc:
            raise ValueError(f"Invalid CSV row {line}: {exc}") from exc

    bars = tuple(bars)
    validate(bars, symbol)
    return Dataset(
        symbol, bars, "raw", source, hashlib.sha256(raw).hexdigest()
    )


def split_chronologically(data, *, validation_start, test_start):
    if data.role != "raw":
        raise ValueError("Split the original raw dataset only.")

    validation_start = utc_time(validation_start)
    test_start = utc_time(test_start)
    if validation_start >= test_start:
        raise ValueError("Validation must start before the test period.")

    groups = {"train": [], "validation": [], "test": []}
    for bar in data.bars:
        role = (
            "train" if bar.time < validation_start
            else "validation" if bar.time < test_start
            else "test"
        )
        groups[role].append(bar)

    if any(not bars for bars in groups.values()):
        raise ValueError("All three chronological partitions must be nonempty.")

    return {
        role: Dataset(
            data.symbol, tuple(bars), role, data.source, data.source_sha256
        )
        for role, bars in groups.items()
    }


def fingerprint(data):
    payload = {
        "symbol": data.symbol,
        "role": data.role,
        "bars": [
            [bar.time.isoformat(),
             *[str(getattr(bar, name)) for name in COLUMNS[1:]]]
            for bar in data.bars
        ],
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
