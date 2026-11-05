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
