import io
import json
from argparse import Namespace
import unittest
from unittest.mock import patch
from helpers import load
b = load('experiments/benchmark.py')
class HttpContractTests(unittest.TestCase):
    def test_request_policy_and_fixed_generation_settings(self):
        for mode in ('baseline','async','inplace','combined'):
            c = b.Client(Namespace(mode=mode))
            request = c.request(2,16,32,'test',1)
            self.assertEqual((len(request['prompt']),request['n_predict'],request['temperature'],request['seed']),(16,32,0,42))
            self.assertEqual(request['adapter_load_policy'],'backend-async' if mode in ('async','combined') else 'sync')
            self.assertEqual(request['id_slot'],1)
    def test_stream_ignores_empty_content_for_ttft(self):
        c = b.Client(Namespace(mode='baseline'))
        request = c.request(1,16,1,'test')
        terminal = dict(stop=True,tokens_evaluated=16,tokens_predicted=1,selected_task_adapter_id=1)
        payload = b''.join(b'data: '+json.dumps(e).encode()+b'\n' for e in ({'content':''},{'content':'x'},terminal))
        with patch.object(c,'http',return_value=io.BytesIO(payload)), patch.object(b,'now_us',side_effect=[1000,3000,5000]):
            response = c.stream(request)
        self.assertEqual(response['ttft_ms'],2)
        self.assertEqual(response['total_ms'],4)
        self.assertEqual(len(response['events']),1)
    def test_wrong_adapter_rejected(self):
        c = b.Client(Namespace(mode='baseline'))
        with self.assertRaisesRegex(RuntimeError,'contract'):
            c.contract(dict(tokens_evaluated=16,tokens_predicted=1,selected_task_adapter_id=2),c.request(1,16,1,'test'))
