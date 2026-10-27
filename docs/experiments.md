# Experiments

The release keeps the request generation, SSE timing, cache preparation, arrival-event schedules and token-gap calculation from the recorded experiments, with paths and server startup separated into the public launcher. [SOURCE_PROVENANCE.json](../experiments/SOURCE_PROVENANCE.json) maps the extracted code to SHA-256 hashes of the original scripts.

The portable client was checked for syntax, command-line behavior and deterministic trace generation. It has **not been rerun on the GPU after packaging**. The CSVs under [paper_results](../experiments/paper_results/) are published measurements, not results produced by the new packaging checks.

## Before running

Prepare the model/adapters and build the runtime using the main README. Run benchmarks on the same Linux machine as the server. The file-page-cache checks use Linux `mincore` and `posix_fadvise`; latency measurements read the local server log.

The default artifact directory is `weights/`, containing `securecode.gguf`, `math.gguf`, `ifeval.gguf`, `bbh.gguf`, `gpqa.gguf`, `mmlupro.gguf` and the model/router artifacts described in the main README. Use `--artifacts /absolute/path` if these six task adapters are stored elsewhere.

Use a fresh server for each configuration and measured throughput run. The `--mode` passed to the client must match the server launcher. `async` and `combined` also set `adapter_load_policy="backend-async"` in every request; changing only the server environment does not enable asynchronous loading. Keep one GPU server running at a time.

```bash
mkdir -p results
python scripts/serve.py --mode baseline --config configs/runtime.json > results/server.log 2>&1
```

Run the client in another terminal. Stop the owned server with Ctrl+C before launching another variant. Each `--out` directory must be new; existing results are never overwritten.

## Serving performance: Figures 8 and 10(a,b)

```bash
# Inspect the request sequence without contacting the server.
python experiments/benchmark.py throughput --mode baseline --rate 0.3 --block 0 --dry-run

# Run one 24-request trace against the already running server.
python experiments/benchmark.py throughput --mode baseline --rate 0.3 --block 0 \
  --out results/throughput/baseline-r03-b0

# After collecting all configurations/rates/blocks:
python experiments/summarize.py results/throughput > results/throughput-summary.json
```

Repeat for `baseline`, `async`, `inplace`, `combined`; rates `0.3`, `0.6`, `0.9`; and blocks `0`, `1`, `2`. This gives 36 runs and 864 requests. Use the same block number across compared configurations.

| Parameter | Value |
|---|---|
| Server slots / task-adapter cache | 2 / 2 |
| Context / batch / microbatch | 2048 / 512 / 512 |
| Requests | 24; eight each of 16, 128 and 256 input tokens |
| Output | 32 tokens, temperature 0, seed 42, ignore EOS |
| Task IDs | 0, 1, 2, 3 in round-robin order |
| Arrival | Fixed intervals `i / rate`; requests dispatch independently of completion |
| Prompt order | `[16,128,256] * 8`, shuffled using `Random(202609230 + block)` |
| File cache | First four task files warmed; task cache prepared by loading IDs 0 through 5 |
| Router / CUDA Graphs / profiler | Explicit IDs bypass router; default Graph policy; no profiler |

The original experiment used a fresh server for every run and a balanced order of modes/rates. Keep the order recorded by `run_latest_throughput_20261014.py` when comparing closely: for block `b`, rotate rates by `b`, then rotate `[baseline,async,inplace,combined]` by `(b + rate_index) % 4`.

Throughput is successful requests divided by elapsed time from the first actual request start to the last response end, including drain. It is a finite-trace measurement. Request latency and TTFT are averaged over 24 requests **within each run**, then medians are taken across the three runs. SLO attainment is the fraction with TTFT at most 6 seconds. Error bars in the paper are minimum/maximum values, not confidence intervals.

## Token-generation delay: Table 5, Figures 9 and 10(c)

`latency` runs one matched-arrival trial. A request on adapter 0 has 16 input and 64 output tokens. Incoming requests use adapter 1, have the selected input length, and generate one token. `repeated` alternates adapters 1 and 2 for six incoming requests. The arrival index counts nonempty HTTP text events; it is not a server token index.

```bash
# Figure 10(c): one ongoing request; 256-token incoming request;
# nonresident adapter, with its file already in the OS page cache.
python experiments/benchmark.py latency --mode baseline --case ablation \
  --state nonresident --input 256 --block 0 --server-log results/server.log \
  --out results/ablation/baseline-M256-b0

# Table 5: six alternating incoming requests, two task-cache entries.
python experiments/benchmark.py latency --mode baseline --case repeated \
  --state nonresident --block 0 --server-log results/server.log \
  --out results/repeated/baseline-b0
```

| Paper output | Client arguments | Server and repetition settings |
|---|---|---|
| Table 5 | `--case repeated --state nonresident --input 16` | 2 slots, cache 2; baseline/combined; blocks 0..29 |
| Fig. 9 | `--case fanout --ongoing 2` or `4`, `--state cold`, inputs 16/128/256 | 3 or 5 slots, context 3072 or 5120, cache 2; baseline/combined; blocks 0..29 |
| Fig. 10(c) | `--case ablation --state resident` or `nonresident`, inputs 16/128/256 | 2 slots, context 2048, cache 2; four configurations; blocks 0..11 |
| Fig. 2(a), Table 1 single | `--case single --state resident` or `cold`, input 16 | 2 slots, cache 2; baseline; blocks 0..29 |
| Table 1 repeated | `--case motivation-repeated --state nonresident --cache 3` or `2` | 2 slots, matching cache capacity, baseline; blocks 0..29; fixed targets 12/20/28/36/44/52 |

For Fig. 9, copy the runtime configuration to a new local file and set `parallel_slots` to 3 or 5 and `context_size` to `1024 * parallel_slots`. For the cache-3 control in Table 1, set `adapter_cache_size` to 3. Keep client `--cache` consistent with that configuration.

The random-arrival experiments use seed 20261012. Single/fanout/ablation select an event in 1..48, shared by matched conditions. Table 5's six events use six sorted samples from 1..18 plus `7*i`, after the 30 fanout draws. The same block selects the same events in every mode.

Max TBT is computed from consecutive **server token-generation timestamps** in `timeline_event=token` records, matched to request labels through `rm_event=slot`. Client HTTP gaps are never substituted. Fanout reports the mean of the individual ongoing requests' Max TBT within a run; the paper reports the median of that value over 30 runs.

### Cold adapter-file reads

Cold runs require the original loader-entry page-residency observer. It checks the specific adapter file immediately before `fopen`; it does not clear the system-wide cache.

```bash
mkdir -p results
gcc -shared -fPIC -O2 -Wall -Wextra -o results/cold_probe.so experiments/cold_probe.c -ldl
LOADING_PROBE_TARGET="$PWD/weights/math.gguf" LD_PRELOAD="$PWD/results/cold_probe.so" \
  python scripts/serve.py --mode baseline --config configs/runtime.json > results/server.log 2>&1
```

The client applies `POSIX_FADV_DONTNEED` only to `math.gguf`, verifies zero resident pages, and requires a zero-page loader-entry observation in the measured log. In the original experiment, device read counters supplied an additional check; that device-specific counter gate is not part of the portable client. `nonresident` means absent from the runtime task-adapter cache while the OS file cache is warm; it is a different condition from `cold`.

## GPU cost profiling: Figure 2(b) and Table 2

[profiling](../experiments/profiling/) preserves the original C++ direct-batch probe, NVTX instrumentation patch/header, fixed prompt and teacher-token plan, Makefile fragment, and Nsight SQLite analysis algorithm. These files came from `lora-cost-20260918-v1`; they profile one ongoing decode token plus 16/128/256 incoming prefill tokens with both adapters resident.

This diagnostic was based on the earlier output-index-corrected runtime, before the final in-place implementation. **The historical instrumentation patch has not been ported or build-tested against the final release runtime.** It is included to document how kernel attribution was performed; applying it to the final checkout is not an automated supported build step.

The original main profile had six runs per length, CUDA Graphs disabled and Nsight Systems 2024.5.4 enabled. It correlated NVTX operator submission ranges with CUDA runtime correlation IDs and kernel intervals. The observed 20.48%, 67.87% and 80.57% ACC shares are fractions of total kernel-duration sums in those traced runs. They are not measured DRAM bandwidth or an overhead-free wall-time breakdown.

If the corresponding SQLite exports are available, arrange them as `PROFILE_RESULTS/main/*-profiled/trace.sqlite` and run:

```bash
python experiments/profiling/analyze_traces.py PROFILE_RESULTS
```

The analysis writes `analysis-traces.json` plus per-trace analysis files under that result directory. It requires the original `LCP/MEASURE/` and `LCP/NODE/` NVTX labels; an arbitrary server trace is not interchangeable.

## Recorded evidence and limits

| Paper output | Original experiment directory |
|---|---|
| Fig. 2(a), Table 1 single | `loading-overhead-random-20261012-v1` |
| Table 1 repeated | `loading-extension-20261008-v1` |
| Fig. 2(b), Table 2 | `lora-cost-20260918-v1` |
| Fig. 8, Fig. 10(a,b) | `throughput-latest-20261014-v1` |
| Table 5, Fig. 9 at input 16 | `evaluation-random-resume-20261012-v1/extension` |
| Fig. 9 at input 128/256 | `evaluation-random-resume-20261012-v1/long` |
| Fig. 10(c) | `evaluation-random-arrival-20261012-v1/four` |

Small plotted-value CSVs are included; model weights, datasets, full response logs and profiler captures are not. The latency CSV also retains supporting completion/TTFT measurements that are not all plotted in the final paper. Its `paired_reduction_percent` column is a median of paired reductions, which can differ from a ratio of marginal medians.

Concurrent output differences remain in the historical records. For example, Table 5 has identical outputs for all requests in 13 of 30 baseline/combined pairs; the four-ongoing-request, 128/256-token groups have zero fully identical pairs out of 30. These observations do not by themselves identify a cause, and performance measurements do not establish output equivalence or task accuracy. The portable client preserves output hashes and server token IDs for independent comparison; it does not mark such differences as resolved.

Packaging changes remove historical string-replacement runners, machine-specific process guards, pre-existing-run gates and device counter paths. The source algorithms and original hashes remain documented, but full historical reproduction also depends on the original artifacts, environment, server-session order and profiling revision. No new model training or serving measurements were run while preparing this release.
