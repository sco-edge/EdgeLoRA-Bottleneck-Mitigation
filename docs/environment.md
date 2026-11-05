# Record an experiment environment

Save a record before each new measurement session, from the environment that
will run that stage:

```bash
mkdir -p results/session-01
python scripts/environment_report.py > results/session-01/environment.json
```

The report contains Python/platform architecture, selected installed package
versions, the repository commit, and whether tracked files differ from it.
A `null` package means it is not installed; `null` Git fields mean Git metadata
could not be read. The report does not import PyTorch or initialize CUDA.

Separately record JetPack/L4T, CUDA compiler and driver versions, GPU model,
power mode, clocks, cooling, storage and available memory. These are deliberately
not inferred from the Python environment. Keep one record per host and stage:
training, conversion and Jetson serving use different environments.

No hostname, username or environment-variable dump is collected. Review any
additional terminal captures before sharing them. The report describes a run's
context; it is neither a dependency lockfile nor evidence of a successful run.
