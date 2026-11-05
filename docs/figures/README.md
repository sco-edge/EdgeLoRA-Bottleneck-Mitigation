# Rebuild the result figures

Run from the repository root in a separate plotting environment:

```bash
python -m pip install -r experiments/requirements-plots.txt
python experiments/plot_results.py serving
python experiments/plot_results.py latency
python experiments/plot_results.py ablation
python experiments/plot_results.py motivation
```

PNG previews and SVG exports are written here; existing generated figures are
replaced. Use `--output-dir results/figures` to preview changes elsewhere.
The script reads the published CSVs and never starts a server. Values and
source rows are validated before plotting. Rendering may vary with fonts or
dependency versions; the numeric inputs are the CSVs, not pixel measurements.

Serving bars show recorded min/max across three runs. The other CSVs provide
medians without dispersion; no error bars are estimated for them. Latency
milliseconds are converted to seconds for readability except in the motivation
loading panel. The captions identify sample counts, units and cache conditions.
