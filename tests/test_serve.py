import copy
import json
import unittest
from helpers import ROOT, load
serve = load('scripts/serve.py')
class ServeTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'configs/runtime.example.json').read_text())
    def test_recorded_commands_for_every_mode(self):
        expected = json.loads((ROOT/'tests/fixtures/serve_commands.json').read_text())
        for mode in serve.MODES:
            with self.subTest(mode=mode):
                actual = serve.make_command(self.config, mode, '127.0.0.1', 8080)[:3]
                normalized = json.loads(json.dumps(actual).replace(str(ROOT).replace('\\', '\\\\'), '<ROOT>'))
                self.assertEqual(normalized, expected[mode])
