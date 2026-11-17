import csv
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal as D, localcontext
from pathlib import Path

from fxagent.backtest import simulate
from fxagent.data import (
    COLUMNS, PIPS, STEP, load_csv, split_chronologically,
)
from fxagent.reporting import save_report


class FoundationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "prices.csv"
        start = datetime(2024, 1, 8, tzinfo=timezone.utc)
        self.rows = [
            {
                "timestamp": (start + index * STEP).isoformat(),
                "open": "1.1", "high": "1.2", "low": "1.0",
                "close": ("1.1", "1.1001", "1.1")[index % 3],
                "spread_pips": "2",
            }
            for index in range(9)
        ]
        self.write_rows(self.rows)
        self.raw = self.load()
        self.parts = split_chronologically(
            self.raw,
            validation_start="2024-01-08T00:45:00Z",
            test_start="2024-01-08T01:30:00Z",
        )

    def write_rows(self, rows):
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def load(self, symbol="EURUSD"):
        return load_csv(
            self.path, symbol,
            source="synthetic unit-test fixture",
            price_basis="mid",
        )

    def simulate_train(self, data=None, **changes):
        settings = {
            "initial_cash": D("10000"),
            "units": 1000,
            "slippage_pips": D("0.5"),
            "commission_per_unit": D("0.0001"),
        }
        settings.update(changes)
        return simulate(
            self.parts["train"] if data is None else data, **settings
        )

    def test_all_five_symbols_load(self):
        for symbol in PIPS:
            with self.subTest(symbol=symbol):
                self.assertEqual(self.load(symbol).symbol, symbol)
        with self.assertRaises(ValueError):
            self.load("BTCUSD")

    def test_bad_values_and_timestamps_are_rejected(self):
        cases = [
            (0, "open", "NaN"),
            (0, "spread_pips", "Infinity"),
            (0, "spread_pips", "-1"),
            (0, "open", "0"),
            (0, "close", "-1"),
            (0, "high", "1.05"),
            (0, "timestamp", "2024-01-08T00:00:00"),
            (0, "timestamp", "2024-01-08T00:01:00Z"),
            (1, "timestamp", self.rows[0]["timestamp"]),
            (1, "timestamp", "2024-01-07T23:45:00Z"),
        ]
        for index, field, value in cases:
            with self.subTest(field=field, value=value):
                rows = [dict(row) for row in self.rows]
                rows[index][field] = value
                self.write_rows(rows)
                with self.assertRaises(ValueError):
                    self.load()

    def test_empty_data_and_wrong_schema_are_rejected(self):
        for text in ("", ",".join(COLUMNS) + "\n", "timestamp,close\n"):
            self.path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                self.load()

    def test_partitions_are_chronological_and_disjoint(self):
        groups = [self.parts[name].bars for name in ("train", "validation", "test")]
        self.assertEqual([len(group) for group in groups], [3, 3, 3])
        self.assertEqual(tuple(bar for group in groups for bar in group), self.raw.bars)
        self.assertLess(groups[0][-1].time, groups[1][0].time)
        self.assertLess(groups[1][-1].time, groups[2][0].time)
        with self.assertRaises(ValueError):
            split_chronologically(
                self.raw,
                validation_start="2024-01-08T00:45:00Z",
                test_start="2024-01-08T00:45:00Z",
            )

    def test_raw_and_test_backtests_are_blocked(self):
        for data in (self.raw, self.parts["test"]):
            with self.assertRaises(ValueError):
                self.simulate_train(data)

    def test_hand_calculated_costs_and_cash(self):
        result = self.simulate_train()
        self.assertEqual(result["number_of_trades"], 1)
        trade = result["trades"][0]
        self.assertEqual(D(trade["entry_price"]), D("1.10015"))
        self.assertEqual(D(trade["exit_price"]), D("1.09985"))
        self.assertEqual(D(trade["spread_cost"]), D("0.20"))
        self.assertEqual(D(trade["slippage_cost"]), D("0.10"))
        self.assertEqual(D(trade["commission_cost"]), D("0.20"))
        self.assertEqual(D(trade["net_pnl"]), D("-0.50"))
        self.assertEqual(D(result["final_cash"]), D("9999.50"))
        self.assertEqual(D(result["bar_close_max_drawdown_fraction"]), D("0.00005"))

    def test_flat_price_with_zero_costs_preserves_cash(self):
        part = self.parts["train"]
        part = replace(part, bars=tuple(
            replace(bar, spread_pips=D("0")) for bar in part.bars
        ))
        result = self.simulate_train(
            part, slippage_pips=D("0"), commission_per_unit=D("0")
        )
        self.assertEqual(D(result["net_pnl"]), D("0"))

    def test_entry_uses_completed_bars_only(self):
        part = self.parts["train"]
        changed_current = replace(
            part,
            bars=part.bars[:-1] + (replace(part.bars[-1], close=D("1.19")),),
        )
        original = self.simulate_train()
        changed = self.simulate_train(changed_current)
        self.assertEqual(changed["number_of_trades"], 1)
        self.assertEqual(
            original["trades"][0]["entry_price"],
            changed["trades"][0]["entry_price"],
        )
        falling_history = replace(
            part,
            bars=(part.bars[0], replace(part.bars[1], close=D("1.09")), part.bars[2]),
        )
        self.assertEqual(self.simulate_train(falling_history)["number_of_trades"], 0)

    def test_adding_future_bars_does_not_change_past_results(self):
        original = self.simulate_train()
        extended = replace(self.parts["train"], bars=self.raw.bars[:6])
        result = self.simulate_train(extended)
        self.assertEqual(original["equity"], result["equity"][:4])
        self.assertEqual(original["trades"][0], result["trades"][0])

    def test_signal_does_not_cross_a_data_gap(self):
        part = self.parts["train"]
        last = replace(part.bars[-1], time=part.bars[-1].time + STEP)
        result = self.simulate_train(replace(part, bars=part.bars[:-1] + (last,)))
        self.assertEqual(result["gaps"], 1)
        self.assertEqual(result["number_of_trades"], 0)

    def test_invalid_settings_and_capital_limit_are_rejected(self):
        for change in (
            {"initial_cash": D("0")},
            {"units": 0},
            {"units": True},
            {"units": 5000},
            {"slippage_pips": D("-1")},
            {"commission_per_unit": D("NaN")},
        ):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    self.simulate_train(**change)

    def test_jpy_pip_size_and_quote_currency(self):
        part = self.parts["train"]
        bars = tuple(
            replace(
                bar,
                open=D("150"), high=D("151"), low=D("149"),
                close=D("150.01") if index == 1 else D("150"),
            )
            for index, bar in enumerate(part.bars)
        )
        result = self.simulate_train(
            replace(part, symbol="USDJPY", bars=bars),
            initial_cash=D("1000000"),
        )
        self.assertEqual(result["account_currency"], "JPY")
        self.assertEqual(D(result["net_pnl"]), D("-30.20"))

    def test_results_are_reproducible_and_precision_is_fixed(self):
        expected = self.simulate_train()
        self.assertEqual(expected, self.simulate_train())
        with localcontext() as context:
            context.prec = 6
            self.assertEqual(expected, self.simulate_train())

    def test_report_records_result_and_fingerprints(self):
        result = self.simulate_train()
        path = save_report(result, self.directory / "reports")
        record = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(record["result"], result)
        self.assertEqual(len(record["package_code_sha256"]), 64)
        self.assertEqual(len(result["source_sha256"]), 64)
        self.assertEqual(len(result["partition_sha256"]), 64)
