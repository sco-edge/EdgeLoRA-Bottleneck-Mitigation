# Historical Nsight analysis inputs

The analyzer expects `RESULTS/main/*-profiled/trace.sqlite`. Each trace must
contain three `LCP/MEASURE/` ranges for incoming lengths 16, 128 and 256 and
the corresponding `LCP/NODE/` ranges. Required core tables include
`StringIds` and `NVTX_EVENTS`; CUDA runtime/driver correlation records connect
node ranges to kernel activities.

Run against a copy of the exported results:

```bash
python experiments/profiling/analyze_traces.py RESULTS
```

The input SQLite connection is read-only. The script writes or replaces
`trace-analysis.json` and `gpu-activities.jsonl` in each trace directory and
`analysis-traces.json` at the results root. Preserve previous analyses if needed.

`complete=true` requires 18 measurements: six runs at each of three lengths.
Missing exports may still produce partial groups; inspect this flag before
interpreting the aggregate. Summed durations count overlapping kernels more
than once; interval unions represent occupied time. CPU and GPU durations
must not be added as disjoint wall-time components.

The instrumentation patch targets an earlier runtime and is not a supported
automatic patch for the final release. See [profiling limits](../../docs/experiments.md).
