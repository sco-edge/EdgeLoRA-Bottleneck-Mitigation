import unittest
from unittest.mock import patch
from helpers import load
env = load('scripts/environment_report.py')
class EnvironmentTests(unittest.TestCase):
    def test_absent_packages_and_git_are_explicit(self):
        with patch.object(env.importlib.metadata,'version',side_effect=env.importlib.metadata.PackageNotFoundError), patch.object(env.subprocess,'check_output',side_effect=FileNotFoundError):
            result = env.report()
        self.assertTrue(all(v is None for v in result['packages'].values()))
        self.assertIsNone(result['repository_commit'])
        self.assertNotIn('hostname',result)
