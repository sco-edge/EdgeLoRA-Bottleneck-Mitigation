# Published aggregate CSVs

These tables preserve manuscript measurements. They are inputs to the
[figure generator](../plot_results.py), not newly executed benchmark outputs.
Some CSVs include a UTF-8 BOM; Python readers should use `utf-8-sig`.

| File | Row identity | Values |
| --- | --- | --- |
| `serving.csv` | mode, offered_rps, metric | min/median/max over three runs |
| `latency.csv` | experiment, condition, metric, configuration | median_ms and optional paired_reduction_percent |
| `motivation.csv` | panel, condition, metric | median, explicit unit and run count |

Serving mode names map as `sync` → Baseline, `async` → Async, `inplace` →
In-place, `async_acc` → Combined. Throughput is requests/s; `latency` and
`ttft` are seconds; `slo` is percent. Offered rate is a workload input, not
the achieved throughput.

In latency conditions, `R` means resident and `M` means nonresident runtime
adapters with warm files. `A2 / 128` means two ongoing requests and 128
incoming input tokens. An empty paired-reduction cell means not provided,
not zero. Paired reductions cannot in general be reconstructed by dividing
the two marginal medians. No per-run spread is provided in this table.

Motivation panel a uses milliseconds; panel b uses percent of summed kernel
durations. These distinct denominators must remain separate in plots.
