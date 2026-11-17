import csv
from datetime import datetime, timezone
from decimal import Decimal as D
from pathlib import Path

from .backtest import simulate
from .data import COLUMNS, STEP, load_csv, split_chronologically
from .reporting import save_report


def main():
    path = Path("data/synthetic_EURUSD_M15.csv")
    path.parent.mkdir(exist_ok=True)
    start = datetime(2024, 1, 8, tzinfo=timezone.utc)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for index in range(96):
            opening = D("1.1000") + D(index) * D("0.00001")
            closing = opening + D(index % 7 - 3) * D("0.0001")
            writer.writerow([
                (start + index * STEP).isoformat(),
                opening,
                max(opening, closing) + D("0.0002"),
                min(opening, closing) - D("0.0002"),
                closing,
                "1.5",
            ])

    raw = load_csv(
        path, "EURUSD",
        source="SYNTHETIC demonstration; not historical market data",
        price_basis="mid",
    )
    partitions = split_chronologically(
        raw,
        validation_start="2024-01-08T12:00:00Z",
        test_start="2024-01-08T18:00:00Z",
    )
    result = simulate(
        partitions["train"],
        initial_cash=D("10000"),
        units=1000,
        slippage_pips=D("0.5"),
        commission_per_unit=D("0.0001"),
    )
    report = save_report(result, "reports")
    print("Synthetic engineering demonstration completed.")
    print("Validation and test partitions were not backtested.")
    print("This demonstration provides no evidence of trading profitability.")
    print(f"Report: {report.resolve()}")


if __name__ == "__main__":
    main()
