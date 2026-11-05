import unittest
from helpers import load
plan = load('experiments/plan_runs.py').plan
class RunOrderTests(unittest.TestCase):
    def test_complete_balanced_order(self):
        rows = plan()
        self.assertEqual(len(rows),36)
        self.assertEqual(len({r['run_name'] for r in rows}),36)
        self.assertEqual([(r['block'],r['rate'],r['mode']) for r in rows[:4]], [(0,.3,m) for m in ('baseline','async','inplace','combined')])
        self.assertEqual((rows[12]['rate'],rows[12]['mode']),(.6,'async'))
        for block in range(3):
            for rate in (.3,.6,.9):
                self.assertEqual({r['mode'] for r in rows if r['block']==block and r['rate']==rate}, {'baseline','async','inplace','combined'})
