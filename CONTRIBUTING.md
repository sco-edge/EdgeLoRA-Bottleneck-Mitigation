# Contributing

Start from a clean checkout and create a branch for one change. Describe the
observed problem, the affected command, and the expected result. For documentation
fixes, identify the command or source that supports the corrected explanation.

Run these CPU checks from the repository root before proposing changes:

```bash
python scripts/check_repository.py
python -m unittest discover -s tests -v
git diff --check
```

Keep paper measurements separate from new runs. Preserve the existing CSVs,
fixed token prompts, source hashes, and runtime patch unless the change explicitly
concerns one of those artifacts. If you change an experiment contract, explain
which historical comparisons remain valid and add a focused regression test.

A passing CPU check is not a CUDA build or a reproduction of GPU measurements.
State the actual hardware and validation performed in the change description.
Do not include downloaded datasets, checkpoints, private logs or credentials.
Third-party notices must stay with their applicable source.

## Sharing an experiment report

Include the public commit, exact command with portable paths, mode, cache/slot
settings, and a minimal error excerpt. For numerical reports, include units,
run counts, aggregation method and whether the run passed all validity gates.
Screenshots of a graph alone are insufficient to reproduce the input values.

Check `git diff --cached --stat` and the staged diff before publishing. Generated
token prompts used by these serving benchmarks are different from downloaded
training data; do not copy prepared JSONL splits into a bug report. Raw responses
and logs can contain source text or local paths even when their filenames look
harmless. Keep credentials and host access details out of reports.

Prefer a small synthetic case that reproduces the issue. Keep third-party
model/data terms separate from the repository's code license, and follow the
existing notices when referencing external artifacts.
