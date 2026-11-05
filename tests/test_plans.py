import json
import unittest
from helpers import ROOT, load
benchmark = load('experiments/benchmark.py')
class PlanTests(unittest.TestCase):
    def test_recorded_traces(self):
        expected = json.loads((ROOT/'tests/fixtures/experiment_plans.json').read_text())
        actual = [benchmark.trace('main', b, 24, r) for b in range(3) for r in (.3,.6,.9)]
        self.assertEqual(actual, expected['traces'])
        for case, plans in expected['arrivals'].items():
            self.assertEqual([benchmark.arrivals(b,case) for b in range(30)], plans)
