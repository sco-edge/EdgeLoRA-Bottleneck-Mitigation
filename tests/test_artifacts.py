import hashlib
from pathlib import Path
import tempfile
import unittest
from helpers import load
verify = load('scripts/verify_artifacts.py').verify
class ArtifactTests(unittest.TestCase):
    def test_transfer_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            file = root/'router.gguf'
            file.write_bytes(b'fixture')
            manifest = {'outputs':{'router.gguf':hashlib.sha256(b'fixture').hexdigest()}}
            self.assertEqual(verify(root,manifest),['router.gguf'])
            file.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'Checksum mismatch'): verify(root,manifest)
            with self.assertRaisesRegex(ValueError,'simple filenames'): verify(root,{'outputs':{'../escape': '0'*64}})
