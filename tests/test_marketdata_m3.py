import csv
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from decimal import Decimal as D
from pathlib import Path

from fxagent.data import load_csv
from fxagent.marketdata_m3 import (import_bidask_ticks, require_quality,
                                   export_m15, TickQualityPolicy)

START = datetime(2024, 1, 8, tzinfo=timezone.utc)


class MarketDataM3Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "input.csv"

    def write_normal(self, bars=3, *, drop=None, spread='0.00010'):
        with self.path.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(('timestamp', 'bid', 'ask'))
            for i in range(bars):
                if i == drop:
                    continue
                for sec, change in ((10, '0'), (450, '0.00020'), (840, '-0.00010')):
                    mid = D('1.1') + D(i) / D(1000) + D(change)
                    s = D(spread)
                    w.writerow(((START + i*timedelta(minutes=15) + timedelta(seconds=sec)).isoformat(),
                                mid - s/2, mid + s/2))

    def convert(self, **kwargs):
        return import_bidask_ticks(self.path, 'EURUSD', provider='SYNTHETIC fixture',
             provenance_url='https://example.invalid/fixture',
             permission_note='Synthetic locally created test data', **kwargs)

    def test_aggregates_three_m15_bars_midpoint_and_spread(self):
        self.write_normal()
        ds, audit = self.convert()
        self.assertEqual(ds.role, 'raw')
        self.assertEqual(len(ds.bars), 3)
        self.assertEqual(ds.bars[0].open, D('1.1'))
        self.assertEqual(ds.bars[0].high, D('1.1002'))
        self.assertEqual(ds.bars[0].low, D('1.0999'))
        self.assertEqual(ds.bars[0].close, D('1.0999'))
        self.assertEqual(ds.bars[0].spread_pips, D('1'))
        self.assertEqual(audit['quote_rows'], 9)
        self.assertTrue(audit['quality_pass'])
        require_quality(audit)

    def test_csv_roundtrip_original_loader(self):
        self.write_normal()
        data, _ = self.convert()
        out = Path(self.tmp.name) / 'roundtrip.csv'
        export_m15(data, out)
        imported = load_csv(out, 'EURUSD', source='roundtrip', price_basis='mid')
        self.assertEqual(data.bars, imported.bars)

    def test_dukascopy_node_epoch_ms_format(self):
        with self.path.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(('timestamp', 'askPrice', 'bidPrice', 'askVolume', 'bidVolume'))
            for i in range(3):
                for sec in (10, 450, 840):
                    moment = START + timedelta(minutes=15*i, seconds=sec)
                    millis = int((moment - datetime(1970,1,1,tzinfo=timezone.utc)).total_seconds()*1000)
                    w.writerow((millis, '1.1001', '1.1000', '1', '1'))
        data, audit = self.convert()
        self.assertEqual(audit['input_format'], 'dukascopy-node-tick')
        self.assertEqual(data.bars[0].open, D('1.10005'))
        self.assertEqual(data.bars[0].spread_pips, D('1'))

    def test_dukascopy_three_column_schema(self):
        with self.path.open('w', newline='') as f:
            w=csv.writer(f); w.writerow(('timestamp','askPrice','bidPrice'))
            for sec in (0,840):
                t=START+timedelta(seconds=sec)
                w.writerow((str(int(t.timestamp()*1000)),'1.1002','1.1000'))
        _, audit = self.convert()
        self.assertTrue(audit['quality_pass'])

    def test_gap_is_reported_and_not_forward_filled(self):
        self.write_normal(bars=4, drop=1)
        ds,a=self.convert()
        self.assertEqual(len(ds.bars),3)
        self.assertEqual(a['missing_m15_intervals_including_market_closures'],1)
        self.assertEqual(a['gap_count'],1)
        self.assertEqual(a['gap_examples'][0]['missing_m15_bars'],1)

    def test_spread_outlier_flags_and_quality_gate(self):
        self.write_normal(spread='0.0012')
        _,a=self.convert()
        self.assertEqual(a['flagged_bars'],3)
        with self.assertRaises(ValueError): require_quality(a)

    def test_flag_sparse_and_late_bars(self):
        self.path.write_text('timestamp,bid,ask\n2024-01-08T00:05:00Z,1.1,1.2\n')
        _,a=self.convert()
        self.assertIn('sparse_quotes',a['flags'][0]['issues'])
        self.assertIn('late_open_quote',a['flags'][0]['issues'])
        self.assertIn('early_last_quote',a['flags'][0]['issues'])

    def test_non_utc_or_naive_rejected(self):
        for timestamp in ('2024-01-08T00:00:00', '2024-01-08T05:30:00+05:30'):
            self.path.write_text(f'timestamp,bid,ask\n{timestamp},1.1,1.2\n')
            with self.assertRaises(ValueError): self.convert()

    def test_duplicate_and_out_of_order_rejected(self):
        for b in ('2024-01-08T00:00:10Z', '2024-01-07T23:59:59Z'):
            self.path.write_text('timestamp,bid,ask\n2024-01-08T00:00:10Z,1.1,1.2\n' + f'{b},1.1,1.2\n')
            with self.assertRaises(ValueError): self.convert()

    def test_zero_negative_crossed_nan_inf_rejected(self):
        for bid,ask in [('0','1.2'),('-1','1.2'),('1.2','1.1'),('NaN','1.2'),('1.1','Infinity')]:
            with self.subTest(bid=bid,ask=ask):
                self.path.write_text(f'timestamp,bid,ask\n2024-01-08T00:00:10Z,{bid},{ask}\n')
                with self.assertRaises(ValueError): self.convert()

    def test_missing_source_provenance_rejected(self):
        self.write_normal()
        with self.assertRaises(ValueError):
            import_bidask_ticks(self.path,'EURUSD',provider='',provenance_url='http://x',permission_note='okay')

    def test_wrong_symbol_or_schema_rejected(self):
        self.write_normal()
        with self.assertRaises(ValueError):
            import_bidask_ticks(self.path,'BTCUSD',provider='p',provenance_url='u',permission_note='note')
        self.path.write_text('timestamp,open,high,low,close\n')
        with self.assertRaises(ValueError): self.convert()

    def test_row_cap_and_policy_validation(self):
        self.write_normal()
        with self.assertRaises(ValueError): self.convert(max_rows=2)
        with self.assertRaises(ValueError): TickQualityPolicy(max_spread_pips=D('NaN'))
        with self.assertRaises(ValueError): TickQualityPolicy(min_quotes_per_bar=1)
        with self.assertRaises(ValueError): TickQualityPolicy(max_open_delay_seconds=900)

    def test_sha_fingerprint_deterministic(self):
        self.write_normal()
        first,a=self.convert()
        second,b=self.convert()
        self.assertEqual(first.source_sha256,second.source_sha256)
        self.assertEqual(a,b)
        self.assertEqual(len(a['input_sha256']),64)

    def test_export_refuses_nonraw(self):
        from dataclasses import replace
        self.write_normal()
        data,_=self.convert()
        with self.assertRaises(ValueError):
            export_m15(replace(data, role='test'), Path(self.tmp.name)/'test.csv')

    def test_jpy_pip_size(self):
        self.path.write_text('timestamp,bid,ask\n'
            '2024-01-08T00:00:10Z,149.99,150.01\n'
            '2024-01-08T00:14:00Z,150.01,150.03\n')
        ds,a=import_bidask_ticks(self.path,'USDJPY',provider='test',provenance_url='u',permission_note='test')
        self.assertEqual(ds.bars[0].spread_pips,D('2'))
        self.assertTrue(a['quality_pass'])


if __name__ == '__main__':
    unittest.main()
