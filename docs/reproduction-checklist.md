# Before comparing a new run with the paper

1. Record the source commit, environment, hardware, power/clock settings and
   model hashes. Distinguish reused historical weights from newly trained ones.
2. Verify the pinned runtime and patch with `scripts/setup_runtime.py --check`.
   Build on the intended Jetson and retain the build output separately.
3. Inspect the launcher dry-run. Confirm the six task paths are in ID order,
   and set the case-specific slots, context and cache capacity.
4. Start one fresh server. Use the matching client mode and a new output folder.
   Run live cache checks on the same Linux machine as the server.
5. Collect the planned blocks and conditions. Preserve invalid/interrupted
   attempts and record any replacement selection.
6. Inspect validity gates, incomplete groups and matched output evidence before
   aggregating. Use the metric units and aggregation levels in [results](results.md).
7. Label new plots with hardware, workload, cache state, repetitions and error-bar
   meaning. Keep newly measured values separate from the published CSVs.

This checklist is a recording aid. Completing it does not guarantee identical
weights, generated outputs or performance. The archived profiling revision and
some original external inputs remain separate reproduction requirements.
