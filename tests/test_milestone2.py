"""Milestone 2 tests: timeframes, signal lookahead, fills, risk, persistence."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D, localcontext
import tempfile
import unittest
from pathlib import Path

from fxagent.data import Bar, Dataset, STEP, fingerprint
from fxagent.timeframes import aggregate_completed, CompletedContext
from fxagent.paper import PaperConfig, simulate_paper, _exit_decision, _price
from fxagent.reporting import save_report

START = datetime(2024, 1, 8, tzinfo=timezone.utc)


def bars_for(n=192, trend='up', spread='1.0'):
    result = []
    for i in range(n):
        level = D('1.10') + D(i) * (D('0.0002') if trend == 'up' else -D('0.0002'))
        close = level + (D('0.00010') if trend == 'up' else -D('0.00010'))
        result.append(Bar(START + STEP * i, level,
                          max(level, close) + D('0.0001'),
                          min(level, close) - D('0.0001'), close, D(spread)))
    return tuple(result)


def dataset(bars=None, role='train', symbol='EURUSD'):
    return Dataset(symbol, bars_for() if bars is None else tuple(bars),
                   role, 'SYNTHETIC unittest only', '0' * 64)


def config(**updates):
    basic = dict(initial_cash=D('10000'), units=1000,
                 stop_pips=D('10'), target_pips=D('25'),
                 slippage_pips=D('0.5'), commission_per_unit=D('0.0001'),
                 max_hold_bars=5)
    basic.update(updates)
    return PaperConfig(**basic)


class TimeframeTests(unittest.TestCase):
    def test_h1_only_full_aligned_groups(self):
        candles = aggregate_completed(bars_for(9), 'EURUSD', 1)
        self.assertEqual(len(candles), 2)
        self.assertEqual(candles[0].start, START)
        self.assertEqual(candles[0].end, START + timedelta(hours=1))
        self.assertEqual(candles[0].high, max(b.high for b in bars_for(4)))
        self.assertEqual(candles[0].low, min(b.low for b in bars_for(4)))

    def test_h4_requires_sixteen_bars(self):
        self.assertEqual(aggregate_completed(bars_for(15), 'EURUSD', 4), ())
        self.assertEqual(len(aggregate_completed(bars_for(16), 'EURUSD', 4)), 1)

    def test_missing_bar_invalidates_window(self):
        bars = bars_for(32)
        missing = bars[:8] + bars[9:]
        self.assertEqual(len(aggregate_completed(missing, 'EURUSD', 4)), 1)
        candles = aggregate_completed(missing, 'EURUSD', 4)
        self.assertEqual(candles[0].start, START + timedelta(hours=4))

    def test_no_early_h4_information(self):
        context = CompletedContext(aggregate_completed(bars_for(40), 'EURUSD', 4), 4)
        self.assertEqual(context.direction_at(START + timedelta(hours=7, minutes=45)), 0)
        self.assertEqual(context.direction_at(START + timedelta(hours=8)), 1)
        self.assertEqual(context.direction_at(START + timedelta(hours=8, minutes=15)), 1)

    def test_incomplete_or_stale_higher_timeframe_is_not_reused(self):
        bars = bars_for(52)
        bars = tuple(b for b in bars if b.time.hour != 8) # missing full H4 block 8-12
        context = CompletedContext(aggregate_completed(bars, 'EURUSD', 4), 4)
        self.assertEqual(context.direction_at(START + timedelta(hours=12)), 0)
        self.assertEqual(context.direction_at(START + timedelta(hours=8)), 1)

    def test_reject_unsupported_hours(self):
        with self.assertRaises(ValueError):
            aggregate_completed(bars_for(16), 'EURUSD', 2)
        with self.assertRaises(ValueError):
            CompletedContext((), 3)

    def test_does_not_depend_on_future_candles(self):
        early = CompletedContext(aggregate_completed(bars_for(40), 'EURUSD', 4), 4)
        later = CompletedContext(aggregate_completed(bars_for(92), 'EURUSD', 4), 4)
        t = START + timedelta(hours=8, minutes=15)
        self.assertEqual(early.direction_at(t), later.direction_at(t))


class PaperTests(unittest.TestCase):
    def test_all_five_supported_symbols_and_quote_currency(self):
        for symbol in ('EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF'):
            base = bars_for(40)
            if symbol == 'USDJPY':
                base = tuple(replace(b, open=b.open*D(130), high=b.high*D(130),
                                     low=b.low*D(130), close=b.close*D(130)) for b in base)
            c = config(initial_cash=D('10000000') if symbol == 'USDJPY' else D('10000'))
            output = simulate_paper(dataset(base, symbol=symbol), c)
            self.assertEqual(output['account_currency'], symbol[-3:])

    def test_train_validation_accepted_raw_test_denied(self):
        for role in ('train', 'validation'):
            self.assertEqual(simulate_paper(dataset(role=role), config())['partition'], role)
        for role in ('raw', 'test'):
            with self.assertRaises(ValueError):
                simulate_paper(dataset(role=role), config())

    def test_sample_rising_and_falling_signals(self):
        bullish = simulate_paper(dataset(), config())
        bearish = simulate_paper(dataset(bars_for(trend='down')), config())
        self.assertGreater(bullish['number_of_trades'], 0)
        self.assertGreater(bearish['number_of_trades'], 0)
        self.assertTrue(all(t['side']=='long' for t in bullish['trades']))
        self.assertTrue(all(t['side']=='short' for t in bearish['trades']))

    def test_every_entry_after_two_completed_h4_candles(self):
        result = simulate_paper(dataset(), config())
        for trade in result['trades']:
            self.assertGreaterEqual(datetime.fromisoformat(trade['entry_time']),
                                    START + timedelta(hours=8))

    def test_lookahead_does_not_affect_closed_past(self):
        first = simulate_paper(dataset(bars_for(48)), config())
        extended = simulate_paper(dataset(bars_for(96)), config())
        self.assertEqual(first['trades'][:-1], extended['trades'][:len(first['trades'])-1])

    def test_spread_slippage_adverse_both_directions(self):
        self.assertEqual(_price(D('1.1000'), D('2'), D('.0001'), D('.00005'), 'buy'), D('1.10015'))
        self.assertEqual(_price(D('1.1000'), D('2'), D('.0001'), D('.00005'), 'sell'), D('1.09985'))

    def test_long_both_stop_target_touch_stop_wins(self):
        p = {'side':'long','stop':D('1.099'),'target':D('1.101')}
        bar = Bar(START, D('1.10'), D('1.102'), D('1.098'), D('1.10'), D('0'))
        self.assertEqual(_exit_decision(p,bar,D('.0001'),D('0')), (D('1.099'), 'stop'))

    def test_short_both_stop_target_touch_stop_wins(self):
        p = {'side':'short','stop':D('1.101'),'target':D('1.099')}
        bar = Bar(START, D('1.10'), D('1.102'), D('1.098'), D('1.10'), D('0'))
        self.assertEqual(_exit_decision(p,bar,D('.0001'),D('0')), (D('1.101'), 'stop'))

    def test_gap_through_stop_executes_worse_open_long(self):
        p = {'side':'long','stop':D('1.099'),'target':D('1.103')}
        bar = Bar(START, D('1.095'), D('1.096'), D('1.094'), D('1.095'), D('0'))
        self.assertEqual(_exit_decision(p,bar,D('.0001'),D('0')), (D('1.095'), 'stop_gap'))

    def test_gap_through_stop_executes_worse_open_short(self):
        p = {'side':'short','stop':D('1.101'),'target':D('1.097')}
        bar = Bar(START, D('1.106'), D('1.107'), D('1.105'), D('1.106'), D('0'))
        self.assertEqual(_exit_decision(p,bar,D('.0001'),D('0')), (D('1.106'), 'stop_gap'))

    def test_spread_and_commission_appear_in_ledger(self):
        r = simulate_paper(dataset(), config())
        t = r['trades'][0]
        self.assertEqual(D(t['net_pnl']), D(t['gross_pnl_after_spread_slippage']) - D(t['commission_total']))
        self.assertEqual(D(t['commission_total']), D('0.20'))
        self.assertEqual(sum((D(t['net_pnl']) for t in r['trades']), D(0)), D(r['net_pnl']))

    def test_oversized_notional_skips_entries(self):
        r = simulate_paper(dataset(), config(units=3000))
        self.assertEqual(r['number_of_trades'], 0)
        self.assertGreater(r['skipped_signals']['risk'], 0)

    def test_oversized_stop_risk_skips_entries(self):
        r = simulate_paper(dataset(), config(stop_pips=D('2000')))
        self.assertEqual(r['number_of_trades'], 0)
        self.assertGreater(r['skipped_signals']['risk'], 0)

    def test_open_position_exits_at_first_observed_price_after_gap(self):
        bars = bars_for(80)
        interrupted = bars[:45] + bars[46:]
        r = simulate_paper(dataset(interrupted),
                           config(max_hold_bars=30, target_pips=D('1000')))
        self.assertIn('gap_liquidation', [t['exit_reason'] for t in r['trades']])

    def test_end_of_data_forces_position_flat(self):
        r = simulate_paper(dataset(bars_for(36)),
                           config(max_hold_bars=30, target_pips=D('1000')))
        self.assertEqual(r['trades'][-1]['exit_reason'], 'end_of_data')
        self.assertEqual(r['open_positions_at_end'], 0)

    def test_daily_realized_loss_gate_prevents_new_entries(self):
        # Synthetic, deliberately enormous spread causes recurring immediate
        # losses; daily gate must stop *new entries* until next UTC day.
        r = simulate_paper(dataset(bars_for(spread='300')), config(max_hold_bars=1))
        self.assertGreater(r['skipped_signals']['daily_loss'], 0)
        self.assertEqual(r['open_positions_at_end'], 0)

    def test_timeouts_and_end_of_data_leave_no_positions(self):
        r = simulate_paper(dataset(), config(max_hold_bars=1, target_pips=D('100')))
        self.assertEqual(r['open_positions_at_end'], 0)
        self.assertTrue(all(t['exit_reason'] in ('timeout','stop','target','stop_gap','end_of_data','gap_liquidation') for t in r['trades']))

    def test_all_equity_rows_and_final_cash_reconcile(self):
        r = simulate_paper(dataset(), config())
        self.assertEqual(len(r['equity']), len(bars_for()))
        self.assertEqual(D(r['final_cash']), D(r['equity'][-1]['cash']))
        self.assertEqual(D(r['final_cash']), D(r['equity'][-1]['equity']))

    def test_invalid_configs(self):
        for change in ({'units': True}, {'initial_cash': D('-1')},
                       {'initial_cash': D('0')}, {'stop_pips': D('0')},
                       {'target_pips': D('NaN')}, {'slippage_pips': D('-1')},
                       {'commission_per_unit': D('Infinity')},
                       {'max_hold_bars': 97}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                simulate_paper(dataset(), config(**change))

    def test_repeatability_with_decimal_precision(self):
        a = simulate_paper(dataset(), config())
        with localcontext() as ctx:
            ctx.prec = 6
            b = simulate_paper(dataset(), config())
        self.assertEqual(a, b)

    def test_report_save_and_digests(self):
        r = simulate_paper(dataset(), config())
        with tempfile.TemporaryDirectory() as temp:
            import json
            saved = save_report(r, Path(temp))
            record = json.loads(saved.read_text())
            self.assertEqual(record['result'], r)
            self.assertEqual(len(r['partition_sha256']), 64)

    def test_signal_cannot_bridge_missing_m15_or_h4_windows(self):
        original = bars_for(64)
        altered = original[:32] + original[33:]
        r = simulate_paper(dataset(altered), config())
        # Missing H4 blocks prevent stale-H4 trade entries after the gap.
        entry_times = [datetime.fromisoformat(t['entry_time']) for t in r['trades']]
        self.assertTrue(all(not (START + timedelta(hours=12) <= t <
                                 START + timedelta(hours=16)) for t in entry_times))

if __name__ == '__main__':
    unittest.main()
