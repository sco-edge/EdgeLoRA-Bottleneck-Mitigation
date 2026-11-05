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

def test_launch_error_is_actionable(self):
    import json
    from unittest.mock import patch
    from helpers import load
    serve = load('scripts/serve.py')
    with patch.object(sys, 'argv', ['serve.py', '--config', str(ROOT/'configs/runtime.example.json')]), patch.object(serve.Path, 'is_file', return_value=True), patch.object(serve.Path, 'mkdir'), patch.object(serve.socket, 'socket'), patch.object(serve.subprocess, 'call', side_effect=PermissionError('permission denied')), patch('sys.stderr') as err:
        with self.assertRaises(SystemExit) as raised: serve.main()
        self.assertEqual(raised.exception.code,2)
        self.assertIn('Cannot execute server', ''.join(c.args[0] for c in err.write.call_args_list))
CliTests.test_launch_error_is_actionable = test_launch_error_is_actionable
