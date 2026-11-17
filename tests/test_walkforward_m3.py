import copy
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

from fxagent.data import Bar, Dataset, STEP
from fxagent.paper import PaperConfig
from fxagent.walkforward_m3 import (Candidate, CostScenario, ResearchPlan,
                                    isolate_pre_holdout, run_walk_forward, DEFAULT_COSTS)
from fxagent.stats_m3 import (descriptive_trade_stats, moving_block_mean_ci,
                              wilson_interval)

START=datetime(2024, 1, 8, tzinfo=timezone.utc)


def make_raw(n=1100):
    bars=[]
    for i in range(n):
        p=D('1.1000') + D(i) * D('.000015')
        close=p + D('.00002')
        bars.append(Bar(START+i*STEP, p, close+D('.0001'), p-D('.0001'), close, D('1.2')))
    return Dataset('EURUSD', tuple(bars), 'raw', 'SYNTHETIC deterministic fixture', 'a'*64)


def candidates():
    return (Candidate('A',PaperConfig(D('10000'),1000,D('8'),D('18'),D('.5'),D('.0001'),5)),
            Candidate('B',PaperConfig(D('10000'),1000,D('10'),D('25'),D('.5'),D('.0001'),5)))


def plan():
    return ResearchPlan(test_start=(START+950*STEP).isoformat(),train_bars=300,
                        validation_bars=160,step_bars=160,embargo_bars=8,
                        min_train_trades=1,min_validation_trades=30)


class WalkForwardTests(unittest.TestCase):
    def setUp(self):
        self.raw=make_raw()
        self.p=plan()
        self.research,self.manifest=isolate_pre_holdout(self.raw,self.p.test_start)

    def test_research_excludes_holdout(self):
        self.assertEqual(len(self.research.bars),950)
        self.assertEqual(self.manifest['holdout_bars_excluded'],150)
        self.assertEqual(self.research.role,'train')
        self.assertLess(self.research.bars[-1].time,datetime.fromisoformat(self.p.test_start))

    def test_holdout_value_changes_do_not_change_research_results(self):
        last=replace(self.raw.bars[-1], close=self.raw.bars[-1].close + D('.0001'),
                     high=self.raw.bars[-1].high + D('.0001'))
        modified=replace(self.raw,bars=self.raw.bars[:-1]+(last,))
        other,_=isolate_pre_holdout(modified,self.p.test_start)
        self.assertEqual(self.research,other)
        self.assertEqual(run_walk_forward(self.research,self.p,candidates()),
                         run_walk_forward(other,self.p,candidates()))

    def test_no_fold_can_cross_holdout(self):
        report=run_walk_forward(self.research,self.p,candidates())
        self.assertEqual(report['test_partition_backtested'],False)
        self.assertGreaterEqual(len(report['folds']),2)
        for f in report['folds']:
            self.assertLess(datetime.fromisoformat(f['validation_end_exclusive']),
                            datetime.fromisoformat(self.p.test_start)+STEP)
            self.assertLess(datetime.fromisoformat(f['train_end_exclusive']),
                            datetime.fromisoformat(f['validation_start']))

    def test_validation_windows_nonoverlapping(self):
        f=run_walk_forward(self.research,self.p,candidates())['folds']
        for left,right in zip(f,f[1:]):
            self.assertLessEqual(left['validation_end_exclusive'],right['validation_start'])

    def test_reject_raw_test_validation_inputs(self):
        for data in (self.raw, replace(self.research,role='test'),
                     replace(self.research,role='validation')):
            with self.assertRaises(ValueError): run_walk_forward(data,self.p,candidates())

    def test_train_only_candidate_selection_does_not_depend_on_future(self):
        data=self.research
        result=run_walk_forward(data,self.p,candidates())
        start=300+8
        altered=[]
        for i,b in enumerate(data.bars):
            if start<=i<start+160:
                # change validation OHLC but not train; preserve valid OHLC
                b=replace(b,close=b.open-D('.00008'),high=b.open+D('.0001'),low=b.open-D('.0001'))
            altered.append(b)
        second=run_walk_forward(replace(data,bars=tuple(altered)),self.p,candidates())
        self.assertEqual(result['folds'][0]['selected'],second['folds'][0]['selected'])
        self.assertEqual(result['folds'][0]['train_candidates'],second['folds'][0]['train_candidates'])

    def test_deterministic_and_cost_scenarios(self):
        one=run_walk_forward(self.research,self.p,candidates())
        two=run_walk_forward(self.research,self.p,candidates())
        self.assertEqual(one,two)
        self.assertEqual(set(one['aggregate_cost_stress']), {'base','spread2_slip2','all_costs3'})
        for f in one['folds']:
            if f['selected']:
                self.assertEqual(f['cost_stress'][0]['net_pnl'],f['validation']['net_pnl'])

    def test_no_selection_if_training_trade_threshold_unmet(self):
        strict=replace(self.p,min_train_trades=100000)
        result=run_walk_forward(self.research,strict,candidates())
        self.assertEqual(result['selected_folds'],0)
        self.assertTrue(all(f['validation'] is None for f in result['folds']))
        self.assertEqual(result['validation_metrics']['trades'],0)

    def test_all_trades_only_validation(self):
        result=run_walk_forward(self.research,self.p,candidates())
        for fold in result['folds']:
            if not fold['validation']: continue
            for trade in fold['validation']['trades']:
                self.assertGreaterEqual(trade['entry_time'],fold['validation_start'])
                self.assertLess(trade['entry_time'],fold['validation_end_exclusive'])

    def test_scenario_restrictions_and_names(self):
        for scenarios in ((CostScenario('base',D(1),D(1),D(1)),
                           CostScenario('base',D(2),D(2),D(2))),
                          (CostScenario('bad',D(1),D(1),D(1)),)):
            with self.assertRaises(ValueError): run_walk_forward(self.research,self.p,candidates(),scenarios)
        with self.assertRaises(ValueError):
            CostScenario('illegal',D('.5'),D('1'),D('1'))

    def test_embargo_must_cover_candidate_holding_horizon(self):
        slow = Candidate('slow', replace(candidates()[0].config, max_hold_bars=20))
        with self.assertRaises(ValueError):
            run_walk_forward(self.research, self.p, (slow,))

    def test_duplicate_candidates_rejected(self):
        with self.assertRaises(ValueError):
            run_walk_forward(self.research,self.p,(candidates()[0], candidates()[0]))

    def test_plan_invalid_window_or_embargo_rejected(self):
        for changed in ({'embargo_bars':7},{'step_bars':159},{'train_bars':63},
                        {'validation_bars':63},{'max_folds':101}):
            with self.subTest(**changed):
                with self.assertRaises(ValueError): replace(self.p,**changed)

    def test_nonexistent_or_empty_holdout_rejected(self):
        for boundary in (START.isoformat(),(START+1200*STEP).isoformat()):
            with self.assertRaises(ValueError): isolate_pre_holdout(self.raw,boundary)

    def test_reject_no_folds(self):
        with self.assertRaises(ValueError):
            run_walk_forward(self.research,replace(self.p,train_bars=900),candidates())

    def test_statistics_wilson(self):
        self.assertIsNone(wilson_interval(0,0))
        lower,upper=wilson_interval(5,10)
        self.assertLess(lower,.5); self.assertGreater(upper,.5)
        with self.assertRaises(ValueError): wilson_interval(3,2)

    def test_stats_zero_and_winning_sample(self):
        blank=descriptive_trade_stats([])
        self.assertEqual(blank['evidence_gate'],'insufficient_trades')
        self.assertIsNone(blank['mean_trade_pnl'])
        self.assertEqual(descriptive_trade_stats([{'net_pnl':'2'}, {'net_pnl':'-1'}], min_required=2)['profit_factor'],'2')

    def test_block_bootstrap_seeded(self):
        data=[D(i) for i in range(14)]
        a=moving_block_mean_ci(data)
        self.assertEqual(a,moving_block_mean_ci(data))
        self.assertLess(a[0],a[1])
        with self.assertRaises(ValueError): moving_block_mean_ci(data,block_size=15)

    def test_original_fingerprint_and_source_persist(self):
        result=run_walk_forward(self.research,self.p,candidates())
        self.assertEqual(result['source_sha256'],self.raw.source_sha256)
        self.assertEqual(result['symbol'],'EURUSD')
        self.assertEqual(result['account_currency'],'USD')
        self.assertEqual(len(result['research_sha256']),64)


if __name__ == '__main__':
    unittest.main()
