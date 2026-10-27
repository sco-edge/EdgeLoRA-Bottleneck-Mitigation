#!/usr/bin/env python3
"""Portable extraction of the paper's HTTP experiments; see SOURCE_PROVENANCE.json.

The fixed trace, request contract, SSE timing, cache preparation and arrival-event
schedules come from the recorded experiment code. Server lifecycle is deliberately
external: start a fresh server with scripts/serve.py for each configuration/run.
"""
import argparse
import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import random
import statistics
import threading
import time
import traceback
import urllib.request

HERE = Path(__file__).resolve().parent
NAMES = ('securecode', 'math', 'ifeval', 'bbh', 'gpqa', 'mmlupro')


def trace(phase, block, n, rps):
    # Preserved from evaluation-rps-20260922-v1/run_rps.py:trace.
    # The final 2026-10-14 study used phase='main', n=24, and blocks 0..2.
    lengths = [16, 128, 256] * (n // 3)
    assert len(lengths) == n
    random.Random((202609220 if phase == 'pilot' else 202609230) + block).shuffle(lengths)
    return [dict(label=f'r{i:03d}', adapter_id=i % 4, input_tokens=lengths[i],
                 n_predict=32, offset_s=i / rps) for i in range(n)]


def arrivals(block, case):
    if case == 'motivation-repeated':
        return [12, 20, 28, 36, 44, 52]
    rng = random.Random(20261012)
    fanout = [rng.randint(1, 48) for _ in range(30)]
    repeated = [tuple(v + 7 * i for i, v in enumerate(sorted(rng.sample(range(1, 19), 6))))
                for _ in range(30)]
    return list(repeated[block]) if case == 'repeated' else [fanout[block]]


def now_us():
    return time.monotonic_ns() // 1000


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


def pages(path):
    """Observe file page residency without touching mapped pages (Linux only)."""
    size = path.stat().st_size
    page = os.sysconf('SC_PAGE_SIZE')
    lib = ctypes.CDLL(None, use_errno=True)
    lib.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    with path.open('rb') as f:
        mapped = mmap.mmap(f.fileno(), size, flags=mmap.MAP_PRIVATE,
                           prot=mmap.PROT_READ | mmap.PROT_WRITE)
        anchor = ctypes.c_char.from_buffer(mapped)
        vector = (ctypes.c_ubyte * ((size + page - 1) // page))()
        try:
            if lib.mincore(ctypes.addressof(anchor), size, vector):
                raise OSError(ctypes.get_errno(), 'mincore failed')
            return dict(resident_pages=sum(v & 1 for v in vector), total_pages=len(vector))
        finally:
            del anchor
            mapped.close()


def warm(path):
    with path.open('rb') as f:
        while f.read(4 << 20):
            pass
    state = pages(path)
    if state['resident_pages'] != state['total_pages']:
        raise RuntimeError(f'Adapter file is not fully page-resident: {path.name}')
    return state


class Client:
    def __init__(self, args):
        self.args = args
        self.prompts = json.loads((HERE / 'prompts.json').read_text())

    def http(self, path, payload=None):
        req = urllib.request.Request(self.args.url.rstrip('/') + path,
              data=None if payload is None else json.dumps(payload).encode(),
              headers={'Content-Type': 'application/json'})
        return urllib.request.urlopen(req, timeout=240)

    def request(self, adapter, length, count, label, slot=None):
        p = dict(prompt=self.prompts[str(length)], adapter_id=adapter, n_predict=count,
                 temperature=0, seed=42, ignore_eos=True, cache_prompt=False,
                 adapter_load_policy='backend-async' if self.args.mode in ('async', 'combined') else 'sync',
                 adapter_load_wait_timeout_ms=120000, experiment_label=label)
        if slot is not None:
            p['id_slot'] = slot
        return p

    @staticmethod
    def contract(response, request):
        if (response.get('tokens_evaluated') != len(request['prompt']) or
            response.get('tokens_predicted') != request['n_predict'] or response.get('truncated') or
            response.get('selected_task_adapter_id') != request['adapter_id']):
            raise RuntimeError('Response failed input/output/adapter contract')

    def complete(self, request):
        with self.http('/completion', request) as handle:
            response = json.load(handle)
        if response.get('error'):
            raise RuntimeError(response['error'])
        self.contract(response, request)
        return response

    def stream(self, request, on_content=None):
        # Extracted from the final base_http.py:stream; retain nonempty SSE timing.
        start = now_us()
        events, terminal = [], None
        with self.http('/completion', dict(request, stream=True)) as handle:
            for line in handle:
                if not line.startswith(b'data: '):
                    continue
                data = line[6:].strip()
                if data == b'[DONE]':
                    continue
                event = json.loads(data)
                if event.get('error'):
                    raise RuntimeError(event['error'])
                if event.get('content'):
                    events.append(dict(mono_us=now_us(), content=event['content']))
                    if on_content:
                        on_content(len(events))
                if event.get('stop'):
                    terminal = event
        end = now_us()
        if terminal is None or not events:
            raise RuntimeError('Missing terminal response or nonempty text event')
        self.contract(terminal, request)
        content = ''.join(e['content'] for e in events)
        return dict(start_us=start, end_us=end, events=events, terminal=terminal,
                    sha256=hashlib.sha256(content.encode()).hexdigest(), payload=request,
                    ttft_ms=(events[0]['mono_us'] - start) / 1000, total_ms=(end - start) / 1000)

    def idle(self, slots):
        for _ in range(600):
            with self.http('/slots') as f:
                active = json.load(f)
            with self.http('/adapter-cache') as f:
                cache = json.load(f)
            if len(active) != slots or cache['capacity'] != self.args.cache:
                raise RuntimeError('Server slot/cache settings do not match experiment')
            if (all(s['state'] == 0 for s in active) and cache['reserved'] == 0 and
                not cache['loader_jobs'] and all(a['ref_count'] == 0 for a in cache['adapters'])):
                return cache
            time.sleep(.05)
        raise RuntimeError('Server did not become idle')

    def clear(self, slots):
        self.idle(slots)
        for slot in range(slots):
            with self.http(f'/slots/{slot}?action=erase', {}) as f:
                json.load(f)
        return self.idle(slots)


def throughput(c, args, out):
    if args.cache != 2:
        raise ValueError('Paper throughput experiment requires cache=2')
    c.clear(2)
    for a in range(6):
        c.complete(c.request(a, 16, 1, 'server-warmup'))
    c.clear(2)
    for a in range(6):
        c.complete(c.request(a, 16, 1, 'prepare'))
    c.clear(2)
    save(out / 'file-cache.json', {n: warm(args.artifacts / f'{n}.gguf') for n in NAMES[:4]})
    rs = trace('main', args.block, 24, args.rate)
    records, threads, lock = {}, [], threading.Lock()
    epoch = now_us() + 200000
    save(out / 'trace.json', dict(epoch_us=epoch, requests=rs))

    def one(r):
        record = dict(r, target_us=epoch + round(r['offset_s'] * 1e6), dispatch_us=now_us())
        try:
            record['response'] = c.stream(c.request(r['adapter_id'], r['input_tokens'], 32, r['label']))
            record['success'] = True
        except Exception:
            record.update(success=False, error=traceback.format_exc())
        with lock:
            records[r['label']] = record
            with (out / 'requests.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    for r in rs:
        delay = (epoch + round(r['offset_s'] * 1e6) - now_us()) / 1e6
        if delay > 0:
            time.sleep(delay)
        thread = threading.Thread(target=one, args=(r,))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    if len(records) != 24 or not all(r['success'] for r in records.values()):
        raise RuntimeError('Request failed; raw records retained')
    if any(not 0 <= r['dispatch_us'] - r['target_us'] < 50000 for r in records.values()):
        raise RuntimeError('Client dispatch exceeded 50 ms lateness gate')
    c.idle(2)
    responses = [r['response'] for r in records.values()]
    elapsed = (max(r['end_us'] for r in responses) - min(r['start_us'] for r in responses)) / 1e6
    return dict(throughput_rps=24 / elapsed,
                mean_latency_s=statistics.mean(r['total_ms'] for r in responses) / 1000,
                mean_ttft_s=statistics.mean(r['ttft_ms'] for r in responses) / 1000,
                slo_percent=100 * sum(r['ttft_ms'] <= 6000 for r in responses) / 24,
                requests=24, elapsed_s=elapsed)


def latency(c, args, out):
    count = args.ongoing
    slots = count + 1
    targets = arrivals(args.block, args.case)
    repeated = args.case in ('repeated', 'motivation-repeated')
    adapters = [1, 2, 1, 2, 1, 2] if repeated else [1]
    c.clear(slots)
    for a in (0, 1, 2):
        c.complete(c.request(a, 16, 1, 'server-warmup', 0))
    if args.case == 'fanout' and args.input in (128, 256):
        for length in (128, 256):
            c.complete(c.request(1, length, 1, f'long-warmup-{length}', 0))
    c.clear(slots)
    sequence = (0, 1, 2, 0) if repeated and args.cache == 3 else (
               (2, 0, 1, 0) if args.state == 'resident' else (1, 2, 0))
    for a in sequence:
        c.complete(c.request(a, 16, 1, 'prepare', 0))
    cache = c.clear(slots)
    expected_b_resident = args.state == 'resident' or (repeated and args.cache == 3)
    if not cache['adapters'][0]['cached'] or cache['adapters'][1]['cached'] != expected_b_resident:
        raise RuntimeError('Unexpected task-adapter cache state')
    files = {}
    for name in ('math', 'ifeval') if repeated else ('math',):
        path = args.artifacts / f'{name}.gguf'
        if args.state == 'cold':
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            files[name] = pages(path)
            if files[name]['resident_pages']:
                raise RuntimeError('Target-file OS page-cache eviction failed')
        else:
            files[name] = warm(path)
    save(out / 'preparation.json', dict(cache=cache, files=files, arrival_targets=targets))
    offset = args.server_log.stat().st_size
    responses, errors, counts, incoming = {}, [], [0] * count, []
    lock = threading.Lock()
    barrier = threading.Barrier(count)
    next_b = 0

    def bwork(j):
        try:
            responses[f'B{j}'] = c.stream(c.request(adapters[j], args.input, 1, f'B{j}', count))
        except Exception:
            errors.append(traceback.format_exc())

    def callback(i, n):
        nonlocal next_b
        with lock:
            counts[i] = n
            if next_b < len(targets) and min(counts) >= targets[next_b]:
                if incoming and incoming[-1].is_alive():
                    raise RuntimeError('Prior incoming request has not completed at next trigger')
                thread = threading.Thread(target=bwork, args=(next_b,))
                incoming.append(thread)
                next_b += 1
                thread.start()

    def awork(i):
        try:
            barrier.wait(timeout=10)
            responses[f'A{i}'] = c.stream(c.request(0, 16, 64, f'A{i}', i),
                                           lambda n: callback(i, n))
        except Exception:
            errors.append(traceback.format_exc())

    ongoing = [threading.Thread(target=awork, args=(i,)) for i in range(count)]
    for thread in ongoing:
        thread.start()
    for thread in ongoing:
        thread.join()
    for thread in incoming:
        thread.join()
    save(out / 'responses.json', dict(responses=responses, errors=errors))
    c.idle(slots)
    with args.server_log.open('rb') as f:
        f.seek(offset)
        text = f.read().decode(errors='replace')
    (out / 'measurement.log').write_text(text, encoding='utf-8')
    if errors or len(responses) != count + len(targets):
        raise RuntimeError('Incomplete latency trial; raw records retained')
    events = []
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    if args.state == 'cold':
        probes = [e for e in events if e.get('cold_probe') == 'before_fopen']
        if not probes or probes[0].get('resident_pages') != 0:
            raise RuntimeError('Cold run needs cold_probe.so: first loader-entry page residency must be zero')
    metrics, tokens = {}, {}
    token_ranges = {}
    for role, response in responses.items():
        slots_log = [e for e in events if e.get('rm_event') == 'slot' and e.get('label') == role]
        if len(slots_log) != 1:
            raise RuntimeError(f'Missing/ambiguous server slot trace: {role}')
        ts = [e for e in events if e.get('timeline_event') == 'token' and
              e.get('id_task') == slots_log[0]['id_task']]
        expected = response['payload']['n_predict']
        if ([e['token_index'] for e in ts] != list(range(1, expected + 1)) or
            any(e['adapter_id'] != response['payload']['adapter_id'] for e in ts)):
            raise RuntimeError(f'Invalid server token trace: {role}')
        tokens[role] = [e['token_id'] for e in ts]
        token_ranges[role] = (ts[0]['mono_us'], ts[-1]['mono_us'])
        metrics[role] = dict(ttft_ms=response['ttft_ms'], total_ms=response['total_ms'],
             max_tbt_ms=max(((b['mono_us'] - a['mono_us']) / 1000 for a, b in zip(ts, ts[1:])), default=None))
    save(out / 'token-ids.json', tokens)
    for i in range(count):
        first, last = token_ranges[f'A{i}']
        if not all(first < responses[f'B{j}']['start_us'] < last for j in range(len(targets))):
            raise RuntimeError('Incoming request did not arrive during every ongoing decode stream')
    return dict(per_request=metrics,
                mean_per_request_max_tbt_ms=statistics.mean(metrics[f'A{i}']['max_tbt_ms'] for i in range(count)),
                mean_ongoing_completion_ms=statistics.mean(metrics[f'A{i}']['total_ms'] for i in range(count)),
                mean_incoming_ttft_ms=statistics.mean(metrics[f'B{i}']['ttft_ms'] for i in range(len(targets))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=('throughput', 'latency'))
    parser.add_argument('--mode', choices=('baseline', 'async', 'inplace', 'combined'), required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--artifacts', type=Path, default=HERE.parent / 'weights')
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--rate', type=float, choices=(.3, .6, .9), default=.3)
    parser.add_argument('--case', choices=('single', 'repeated', 'motivation-repeated', 'fanout', 'ablation'), default='single')
    parser.add_argument('--state', choices=('resident', 'nonresident', 'cold'), default='nonresident')
    parser.add_argument('--input', type=int, choices=(16, 128, 256), default=16)
    parser.add_argument('--ongoing', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--cache', type=int, choices=(2, 3), default=2)
    parser.add_argument('--server-log', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Print the exact trace/arrival plan without HTTP or file writes')
    args = parser.parse_args()
    if not 0 <= args.block < (3 if args.experiment == 'throughput' else 30):
        parser.error('block must be 0..2 for throughput or 0..29 for latency')
    if args.case == 'ablation' and args.block >= 12:
        parser.error('ablation uses blocks 0..11')
    if args.experiment == 'latency':
        if args.case in ('single', 'ablation', 'repeated', 'motivation-repeated') and args.ongoing != 1:
            parser.error('single, ablation and repeated cases use one ongoing request')
        if args.case in ('repeated', 'motivation-repeated') and (args.input != 16 or args.state == 'cold'):
            parser.error('repeated loading uses 16-token inputs and warm OS page cache')
        if args.case == 'ablation' and args.state == 'cold':
            parser.error('paper ablation prewarms adapter files; use resident/nonresident')
    plan = trace('main', args.block, 24, args.rate) if args.experiment == 'throughput' else dict(
           targets=arrivals(args.block, args.case), ongoing=args.ongoing, incoming_tokens=args.input,
           slots=args.ongoing + 1, cache=args.cache, state=args.state)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if args.out is None or (args.experiment == 'latency' and args.server_log is None):
        parser.error('--out is required; latency also requires --server-log')
    args.out.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    save(args.out / 'config.json', config)
    save(args.out / 'plan.json', plan)
    status = dict(valid=False)
    try:
        client = Client(args)
        status['metrics'] = throughput(client, args, args.out) if args.experiment == 'throughput' else latency(client, args, args.out)
        status['valid'] = True
    except Exception:
        status['error'] = traceback.format_exc()
        raise
    finally:
        save(args.out / 'summary.json', status)


if __name__ == '__main__':
    main()
