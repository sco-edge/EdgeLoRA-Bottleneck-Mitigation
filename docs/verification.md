# Offline verification

From the repository root, using Python 3.10 or newer:

```bash
python scripts/check_repository.py
python -m unittest discover -s tests -v
python scripts/serve.py --config configs/runtime.example.json --mode combined --dry-run
python experiments/benchmark.py throughput --mode baseline --dry-run
```

Repository checks parse tracked Python without importing training libraries,
parse JSON, and check local Markdown file targets. They do not fetch remote
links or validate section anchors. Untracked files are outside their scope.

Regression tests compare all four server commands with preserved fixtures,
freeze deterministic traces/arrivals, exercise SSE/request contracts, and
check aggregation, artifact integrity and input errors using synthetic data.
No model download, server connection or CUDA execution is required.

Dry-run validates the request plan and configuration construction. It does not
establish that weights are present or that a GPU run will succeed. The fixed
token IDs in the benchmark prompts depend on the recorded tokenizer/model.
