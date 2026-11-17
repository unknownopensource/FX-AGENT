"""Completed H1/H4 aggregates from UTC M15 midpoint bars (offline only).

A higher-timeframe candle is emitted only when all M15 source bars are present
inside the aligned UTC window. Entries never use a candle before its close.
"""
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from .data import STEP, Bar, validate


@dataclass(frozen=True)
class CompletedCandle:
    start: datetime
    end: datetime              # exclusive boundary, becomes knowable at end
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal


def _boundary(time: datetime, hours: int) -> datetime:
    if time.tzinfo is None or time.utcoffset() != timedelta(0):
        raise ValueError("UTC-aware bar times required")
    return time.replace(hour=(time.hour // hours) * hours,
                        minute=0, second=0, microsecond=0)


def aggregate_completed(bars: tuple[Bar, ...], symbol: str, hours: int) -> tuple[CompletedCandle, ...]:
    """Use ONLY full aligned H1/H4 groups; never fill missing bars."""
    if hours not in (1, 4):
        raise ValueError("Only H1 and H4 supported")
    validate(bars, symbol)
    expected = hours * 4
    groups = {}
    for bar in bars:
        start = _boundary(bar.time, hours)
        groups.setdefault(start, []).append(bar)
    output = []
    for start, group in sorted(groups.items()):
        if (len(group) != expected or group[0].time != start
                or any(group[i].time != start + i * STEP for i in range(expected))):
            continue
        output.append(CompletedCandle(
            start, start + timedelta(hours=hours), group[0].open,
            max(b.high for b in group), min(b.low for b in group), group[-1].close
        ))
    return tuple(output)


class CompletedContext:
    """Queries select candles fully closed at (or before) the M15 entry time.

    Rejects stale candles and trends separated by incomplete HTF windows.
    """
    def __init__(self, candles: tuple[CompletedCandle, ...], hours: int):
        if hours not in (1, 4):
            raise ValueError("Only H1/H4 supported")
        self.candles = candles
        self.hours = hours
        self.ends = tuple(c.end for c in candles)

    def direction_at(self, entry_time: datetime) -> int:
        """Return +1/-1 only for two fresh, consecutive, CLOSED HTF closes."""
        boundary = _boundary(entry_time, self.hours)
        index = bisect_right(self.ends, entry_time) - 1
        if index < 1:
            return 0
        current, previous = self.candles[index], self.candles[index - 1]
        if current.end != boundary or previous.end != current.start:
            return 0
        return (current.close > previous.close) - (current.close < previous.close)
