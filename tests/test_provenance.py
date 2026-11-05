import hashlib
import json
import unittest
from helpers import ROOT
class ProvenanceTests(unittest.TestCase):
    def test_runtime_patch_checksum(self):
        p = ROOT/'patches/runtime_manifest.json'
        manifest = json.loads(p.read_text())
        self.assertEqual(hashlib.sha256((p.parent/manifest['patch_file']).read_bytes()).hexdigest(),manifest['patch_sha256'])
        self.assertEqual(len(manifest['files']),14)
    def test_renamed_router_sources_retain_archived_hashes(self):
        manifest = json.loads((ROOT/'training/configs/provenance.json').read_text())
        mapping = {'train_six_label_router_smoke.py':'train_router.py', 'prepare_six_label_router_full_v2.py':'prepare_router_full_dataset.py', 'prepare_six_label_router_profile_groups.py':'prepare_router_profile_groups.py'}
        for old,new in mapping.items():
            digest = hashlib.sha256((ROOT/'training/adapter-router'/new).read_bytes()).hexdigest()
            self.assertEqual(digest,manifest['source_file_sha256']['adapter-router/'+old])
