"""Synthetic Milestone 2 engineering demo. Never submits orders."""
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from pathlib import Path
from .data import COLUMNS, load_csv, split_chronologically
from .paper import PaperConfig, simulate_paper
from .reporting import save_report


def main():
    path = Path('data/synthetic_EURUSD_M15_m2.csv')
    path.parent.mkdir(exist_ok=True)
    first = datetime(2024, 1, 8, tzinfo=timezone.utc)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for i in range(192):
            # Deterministic artificially trending prices; NOT market history.
            midpoint = D('1.1000') + D(min(i, 100)) * D('0.00004') - D(max(0, i - 100)) * D('0.00005')
            closing = midpoint + (D('0.00002') if i < 100 else -D('0.00002'))
            writer.writerow([(first + timedelta(minutes=15*i)).isoformat(),
                             midpoint, max(midpoint, closing) + D('0.00005'),
                             min(midpoint, closing) - D('0.00005'), closing,
                             '1.2'])
    raw = load_csv(path, 'EURUSD', source='SYNTHETIC M2 engineering demo', price_basis='mid')
    sections = split_chronologically(raw,
        validation_start='2024-01-09T08:00:00Z',
        test_start='2024-01-09T16:00:00Z')
    config = PaperConfig(initial_cash=D('10000'), units=1000,
                         stop_pips=D('12'), target_pips=D('18'),
                         slippage_pips=D('0.5'), commission_per_unit=D('0.0001'),
                         max_hold_bars=8)
    result = simulate_paper(sections['train'], config)
    report = save_report(result, 'reports')
    print('M2 synthetic paper research demonstration complete.')
    print('Trades:', result['number_of_trades'])
    print('Quote-currency net P&L:', result['net_pnl'], result['account_currency'])
    print('Report:', report.resolve())
    print('No market data, no broker, no AI API, no test-partition backtest.')


if __name__ == '__main__':
    main()
