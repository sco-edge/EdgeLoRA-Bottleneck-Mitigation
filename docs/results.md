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
