import subprocess
import sys
import unittest
from helpers import ROOT
class CliTests(unittest.TestCase):
    def test_missing_config_has_actionable_error(self):
        r = subprocess.run([sys.executable, str(ROOT/'scripts/serve.py'), '--config', str(ROOT/'tests/absent.json'), '--dry-run'], capture_output=True, text=True)
        self.assertEqual(r.returncode,2)
        self.assertIn('Cannot load runtime config',r.stderr)
        self.assertNotIn('Traceback',r.stderr)
