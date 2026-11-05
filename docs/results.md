# Portable experiment result files

Each `--out` directory represents one attempt. The client creates it with
overwrite protection and writes `config.json` and `plan.json` before running.

| File | Applies to | Purpose |
| --- | --- | --- |
| `config.json` | Both | Resolved CLI options, including mode and block |
| `plan.json` | Both | Planned request sequence or arrival conditions |
| `summary.json` | Both | `valid`, metrics on success, traceback on failure |
| `trace.json` | Throughput | Epoch and all scheduled requests |
| `requests.jsonl` | Throughput | Completed request records; completion order may differ from labels |
| `file-cache.json` | Throughput | Adapter file-page residency after warming |
| `preparation.json` | Latency | Prepared runtime cache, file pages and triggers |
| `responses.json` | Latency | Streams and captured request errors |
| `measurement.log` | Latency | Server-log segment after the preparation offset |
| `token-ids.json` | Latency | Validated server token sequences by request role |

A forcefully terminated process may leave no summary. The aggregator exposes
such directories under `incomplete_runs` when `config.json` exists. Runs with
`valid=false` appear under `invalid_runs`. A failure may occur before later
files in the table are written, so their absence is not proof of deletion.

## Metric units and aggregation

| Metric | Unit | Definition |
| --- | --- | --- |
| `throughput_rps` | requests/s | 24 divided by first actual start to last end |
| `mean_latency_s` | seconds | Mean client completion time across 24 requests |
| `mean_ttft_s` | seconds | Mean time to first nonempty text event |
| `slo_percent` | percent | Fraction of requests with TTFT <= 6 seconds |
| `mean_per_request_max_tbt_ms` | milliseconds | Mean of ongoing requests' individual server-token maximum gaps |
| `mean_ongoing_completion_ms` | milliseconds | Mean completion time of ongoing requests |
| `mean_incoming_ttft_ms` | milliseconds | Mean TTFT of incoming requests |

The summary tool aggregates top-level numeric metrics over runs using median,
minimum and maximum. Nested `per_request` data remains in each raw summary.
It does not pool all request latencies into a single median or estimate a
confidence interval. Compare like units before calculating changes.

`max_tbt_ms` is `null` for a one-token request because no token pair exists.
This is not zero delay. The client uses monotonic timestamps; they are not
calendar times and should not be aligned across different machine clocks.

## Matched repetitions and incomplete comparisons

```bash
python experiments/summarize.py results/throughput --require-blocks 0 1 2
```

This optional gate requires exactly those blocks in every observed group.
It does not prove that all twelve mode/rate groups exist; compare the observed
group set with the planned 36 runs. Invalid and interrupted runs remain listed
separately even if successful replacement runs satisfy the block requirement.

A duplicate block in one configuration raises an error. Keep alternate attempts
outside the aggregation root, or select one attempt per block in a new analysis
directory while preserving all originals. Document the replacement criterion.
Do not select attempts by which result looks faster.

For a paired percentage, match block and condition first, compute each pair's
change, then apply the intended aggregation. The median of paired percentages
need not equal the percentage change between separately computed medians.
