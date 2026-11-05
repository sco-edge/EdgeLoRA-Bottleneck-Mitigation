from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from helpers import load
b = load('experiments/benchmark.py')
class PreflightTests(unittest.TestCase):
    def test_missing_artifacts_and_supported_host(self):
        with tempfile.TemporaryDirectory() as d:
            a = Namespace(experiment='throughput', cache=2, url='http://localhost:8080', artifacts=Path(d))
            with patch.object(b.sys,'platform','linux'):
                with self.assertRaisesRegex(ValueError,'Missing or empty'): b.validate_inputs(a)
                for name in b.NAMES[:4]: (Path(d)/f'{name}.gguf').write_bytes(b'fixture')
                b.validate_inputs(a)
            with patch.object(b.sys,'platform','win32'):
                with self.assertRaisesRegex(ValueError,'Linux'): b.validate_inputs(a)
