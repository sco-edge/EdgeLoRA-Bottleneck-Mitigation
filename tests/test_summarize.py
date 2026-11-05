import json
from pathlib import Path
import tempfile
import unittest
from helpers import load
summarize = load('experiments/summarize.py')
class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
    def run_record(self, name, block, value, valid=True):
        p = self.root/name
        p.mkdir()
        (p/'config.json').write_text(json.dumps(dict(experiment='throughput', mode='baseline', block=block)))
        (p/'summary.json').write_text(json.dumps(dict(valid=valid, metrics=dict(throughput_rps=value))))
        return p
    def test_aggregation_and_invalid_runs(self):
        for i,v in enumerate([2,9,4]): self.run_record(str(i),i,v)
        self.run_record('failed',3,0,False)
        result = summarize.aggregate(self.root)
        self.assertEqual(result['groups'][0]['metrics']['throughput_rps'], dict(median=4, minimum=2, maximum=9))
        self.assertEqual(result['groups'][0]['blocks'], [0,1,2])
        self.assertEqual(len(result['invalid_runs']),1)
    def test_duplicate_blocks_rejected(self):
        self.run_record('first',0,2)
        self.run_record('rerun',0,3)
        with self.assertRaisesRegex(ValueError,'Duplicate block'): summarize.aggregate(self.root)

def test_nonfinite_values_rejected(self):
    self.run_record('bad',0,float('nan'))
    with self.assertRaisesRegex(ValueError,'invalid numeric'): summarize.aggregate(self.root)
SummaryTests.test_nonfinite_values_rejected = test_nonfinite_values_rejected
def test_metric_schema_mismatch(self):
    self.run_record('one',0,1)
    p = self.run_record('two',1,2)
    (p/'summary.json').write_text(json.dumps(dict(valid=True,metrics=dict(other=3))))
    with self.assertRaisesRegex(ValueError,'Inconsistent numeric'): summarize.aggregate(self.root)
SummaryTests.test_metric_schema_mismatch = test_metric_schema_mismatch

def test_interrupted_run_is_visible(self):
    p = self.run_record('interrupted',0,2)
    (p/'summary.json').unlink()
    self.assertEqual(summarize.aggregate(self.root)['incomplete_runs'], ['interrupted'])
SummaryTests.test_interrupted_run_is_visible = test_interrupted_run_is_visible

def test_required_blocks(self):
    self.run_record('one',0,2)
    with self.assertRaisesRegex(ValueError,'Incomplete block'): summarize.aggregate(self.root,[0,1,2])
    self.assertEqual(summarize.aggregate(self.root,[0])['groups'][0]['n'],1)
SummaryTests.test_required_blocks = test_required_blocks
