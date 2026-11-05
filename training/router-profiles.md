# Router profile inputs

The two preparation scripts serve different stages. They are preserved
experimental procedures, not a one-command downloader for a new dataset.

| Script | Input | Output |
| --- | --- | --- |
| `adapter-router/prepare_router_profile_groups.py` | Prepared validation rows and frozen exclusion groups | Deterministic prompt groups shard003..shard013 |
| `adapter-router/prepare_router_full_dataset.py` | Completed adapter cross-profiles | Frozen train/validation router JSONL and metadata |

The profile-group script contains relative `adapter-training/...` paths. When
replaying that archived layout, run it from `training/`. Review `SOURCE_PATHS`
and the three frozen exclusion directories first. The full-dataset script
accepts `--evaluations-root`, repeatable `--profiling-dir GROUP=PATH`, and
`--repo-root` for resolving the recorded inputs.

Each completed profile must contain `cross_predictions.jsonl`,
`routing_labels.jsonl`, `summary.json`, and `run_config.json`. Training uses
shard000, shard001 and shard003..shard009; validation uses shard010..shard011.
shard002 is excluded, and shard012..shard013 remain reserved for independent
testing. Directory names, metadata dates and hashes are checked by the source.

A fresh single cross-profile is not interchangeable with this full set. Keep
held-out groups separate and retain failures instead of editing metadata to
satisfy the freezer. See [training instructions](README.md).
