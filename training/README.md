# Training and model preparation

The serving experiments ran on Jetson. A separate RTX 2080 Ti host trained
the six task adapters and the six-label router. The base model and training
datasets came from Hugging Face. The deployed task adapters were **locally
trained PEFT checkpoints**, then converted to F32 GGUF; they were not six
ready-made adapters downloaded from Hugging Face.

This directory contains the training, data preparation, cross-profiling, and
conversion code. Weights, datasets, generated answers, environment folders,
and intermediate checkpoints are excluded.

## What is here

| Path | Purpose |
| --- | --- |
| `adapter-training/train_task_adapter.py` | Train one task adapter with assistant-only loss |
| `adapter-training/prepare_*.py` | Prepare answer-bearing task data |
| `adapter-training/download_secucoder.py` | Download the recorded SecuCoder revision in the dataset-on-disk format expected by its preparation script |
| `adapter-training/cross_profile_task_adapters.py` | Evaluate each prompt with all six adapters to construct router labels |
| `adapter-training/evaluate_task_adapter.py` | Generation and task-specific scoring used by cross-profiling |
| `adapter-router/prepare_six_label_router_full_v2.py` | Freeze the recorded training/validation profile groups |
| `adapter-router/train_six_label_router_smoke.py` | Train the router; despite its historical filename, this also ran the full 3,410-step model |
| `convert_deployment_artifacts.py` | Export task/router GGUF files and the separate router classifier |
| `configs/` | Recorded settings, input fingerprints, and model/dataset provenance |

## Environment and base model

Use a separate Linux Python environment on the training GPU host:

```bash
python3 -m venv .venv-training
source .venv-training/bin/activate
pip install -r training/requirements-rtx2080ti.txt
```

The recorded training configuration used PyTorch `2.4.1+cu121`, Transformers
`4.45.2`, and PEFT `0.13.2`. The requirements file is the archived RTX
requirements, rather than a later environment snapshot. GPU training has not
been rerun as part of packaging this release.

Obtain access to [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
under its upstream terms, authenticate locally, and download revision
`0cb88a4f764b7a12671c53f0838cd831a0843b95`. Set `BASE_MODEL` to that local
snapshot. Do not commit authentication files or downloaded weights.

From the repository root, download the required Safetensors and tokenizer
files into the ignored `models/` directory:

```bash
# This Hub version is a release constraint, not a recovered training lockfile.
python -m pip install "huggingface_hub==0.25.2"
huggingface-cli login
export BASE_MODEL="$PWD/models/Llama-3.2-3B-Instruct"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    revision="0cb88a4f764b7a12671c53f0838cd831a0843b95",
    local_dir=os.environ["BASE_MODEL"],
    allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
    ignore_patterns=["original/*"],
)
PY
```

These options follow the [versioned Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/v0.25.2/guides/download).
The requirements file pins the main training libraries; it is not a complete
lockfile for every transitive dependency. `datasets` supplies PyArrow for the
Parquet/Arrow preparation readers.

## Task adapters

The runtime IDs are fixed. Their names are routing labels, not claims that
the identically named evaluation benchmarks were used as training datasets.

| ID | Label | Recorded public data sources |
| --- | --- | --- |
| 0 | `securecode` | `scthornton/securecode`, `ivitopow/SecuCoder` |
| 1 | `math` | `TIGER-Lab/MathInstruct` |
| 2 | `ifeval` | `HuggingFaceH4/no_robots` |
| 3 | `bbh` | `kaist-ai/CoT-Collection` |
| 4 | `gpqa` | `allenai/ai2_arc`, `allenai/openbookqa`, `allenai/sciq` |
| 5 | `mmlupro` | `Rowan/hellaswag`, `allenai/winogrande`, `tau/commonsense_qa` |

Known dataset revisions are in [configs/dataset-revisions.json](configs/dataset-revisions.json).
The audit did not establish pinned revisions for `scthornton/securecode` and
`TIGER-Lab/MathInstruct`. Each dataset retains its own terms; for example, the
archived SecuCoder downloader records `CC-BY-NC-SA-4.0`. No dataset content is
redistributed here.

Preparation entry points:

```bash
# Each command accepts explicit input/output paths; inspect --help first.
python training/adapter-training/prepare_securecode_math_data.py --help
python training/adapter-training/prepare_securecode_v2.py --help
python training/adapter-training/prepare_task_adapter_data.py --help
```

The included SecuCoder downloader obtains its pinned revision without needing
a separate dataset-layout conversion:

```bash
python training/adapter-training/download_secucoder.py \
  --output training/adapter-training/raw/secucoder
```

The first script handles SecureCode/MathInstruct, the second contains the
later SecureCode preparation, and the third prepares the other four tasks.
`prepare_task_adapter_data.py` expects the pinned source revisions recorded
in `raw/revisions.json`; use the corresponding entries from
`configs/dataset-revisions.json`. The source loaders define the expected
JSON, Parquet, and dataset-on-disk layouts. Preserve the tokenizer validation
when preparing data.

For a prepared root containing `<task>/train.jsonl` and
`<task>/validation.jsonl`, train one adapter with:

```bash
python training/adapter-training/train_task_adapter.py \
  --model "$BASE_MODEL" --task math \
  --data-root "$TASK_DATA_ROOT" \
  --output-root training/adapter-training/outputs
```

Repeat with the appropriate prepared root for each task. The script uses
rank 16, alpha 32, dropout 0.05, cutoff 512, seed 42, two epochs, microbatch
1, gradient accumulation 16, and learning rate `2e-4`. It writes
`outputs/full/<task>/`. Recorded row counts and settings are in
[configs/task-adapters.json](configs/task-adapters.json).

## Six-label router

The router is a LoRA sequence classifier with six multi-hot output labels.
Its labels came from cross-profiling the task adapters, not directly from
the source task name. The released full-data preparation script keeps rows
with at least one exact positive and excludes weak-only labels from default
router training. The held-out groups remain separate.

The archived preparation scripts retain their original group contracts
(`shard000`/`shard001`, `shard003`-`shard011`) and date checks. They are
provided to expose the experimental procedure; the frozen profiles and
prepared JSONL datasets are not bundled. Reconstructing the exact prepared
data byte-for-byte requires those external inputs. The current source does
not by itself establish every historical preprocessing-version choice.

Cross-profiling accepts `--model "$BASE_MODEL"`, six `--data-root TASK=PATH`
arguments, and `--adapters-root training/adapter-training/outputs/full`.
Each task data root contains a held-out `test.jsonl`. Use
`--max-new-tokens 1024` for the recorded full-v2 profile contract; the historical CLI default
is 256. For example, after defining `PROFILE_ROOT` with the six task folders:

```bash
python training/adapter-training/cross_profile_task_adapters.py \
  --model "$BASE_MODEL" \
  --adapters-root training/adapter-training/outputs/full \
  --data-root "securecode=$PROFILE_ROOT/securecode" \
  --data-root "math=$PROFILE_ROOT/math" \
  --data-root "ifeval=$PROFILE_ROOT/ifeval" \
  --data-root "bbh=$PROFILE_ROOT/bbh" \
  --data-root "gpqa=$PROFILE_ROOT/gpqa" \
  --data-root "mmlupro=$PROFILE_ROOT/mmlupro" \
  --split test --max-samples 32 --batch-size 1 --max-new-tokens 1024 \
  --output training/adapter-training/evaluations/cross-profile/new-profile
```

This produces a new profile. The archived full-v2 freezer additionally
requires the recorded set of profile groups and their metadata; a single
new profile is not the full training dataset.

To replay training with the prepared dataset, set `ROUTER_DATA` to a directory
containing `train.jsonl`, `validation.jsonl`, `stats.json`, and `manifest.json`:

```bash
python training/adapter-router/train_six_label_router_smoke.py \
  --base-model "$BASE_MODEL" --data-dir "$ROUTER_DATA" \
  --output-dir training/adapter-router/trained_models/six_label_router_full_v2_e10_v1 \
  --max-steps 3410 --train-samples 1364 --validation-samples 302 \
  --micro-batch-size 1 --gradient-accumulation-steps 4 \
  --learning-rate 1e-5 --cutoff-len 512 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05
```

The output directory must not already exist. The script verifies that the
saved and reloaded classifier tensors match. See
[configs/router-full-v2-e10.json](configs/router-full-v2-e10.json) for the
recorded dataset hashes and settings. Retraining is not a guarantee of
bit-identical weights or identical serving results.

## Convert for Jetson

Prepare the patched runtime with `python scripts/setup_runtime.py` first.
Perform Python conversion in a separate environment on the Linux training
host. The upstream conversion requirements request PyTorch 2.2.x and
Protobuf below 5, which differ from the training environment:

```bash
python3 -m venv .venv-convert
source .venv-convert/bin/activate
python -m pip install \
  -r vendor/EdgeLoRA/edgelora/requirements/requirements-convert_lora_to_gguf.txt \
  "transformers==4.45.2" "huggingface_hub==0.25.2" "safetensors==0.4.5"
python -m pip install --no-deps ./vendor/EdgeLoRA/edgelora/gguf-py
```

Use Python 3.10 or 3.11 for these older wheel versions. The local `gguf-py`
package is the version bundled with the pinned runtime. The following
commands convert the base snapshot to F16 GGUF, then quantize it to Q4_0:

```bash
mkdir -p weights
python vendor/EdgeLoRA/edgelora/convert_hf_to_gguf.py "$BASE_MODEL" \
  --outfile weights/base-f16.gguf --outtype f16

# CPU quantizer build on a fresh training-host checkout; requires make and a C++ compiler.
make -C vendor/EdgeLoRA/edgelora -j1 llama-quantize
vendor/EdgeLoRA/edgelora/llama-quantize \
  weights/base-f16.gguf weights/base-q4_0.gguf Q4_0
```

Run these commands with new output filenames if the weights already exist.
On Jetson, `bash scripts/build_runtime.sh` already builds the quantizer with
the same CUDA flags as the serving binary; use that binary instead of mixing
CPU and CUDA object files in one build directory. The F16 file is an
intermediate artifact and is not needed for serving.

In the conversion environment, export the adapters using the patched
converter's `--router-classifier-out` support:

```bash
python training/convert_deployment_artifacts.py \
  --runtime-dir vendor/EdgeLoRA/edgelora --base "$BASE_MODEL" \
  --task-adapters training/adapter-training/outputs/full \
  --router-adapter training/adapter-router/trained_models/six_label_router_full_v2_e10_v1 \
  --output-dir weights
```

The runtime directory contains `convert_lora_to_gguf.py` inside the patched
EdgeLoRA checkout. The wrapper exports seven F32 GGUF
adapters and `router-classifier.bin`, verifies the classifier against the
PEFT checkpoint, and writes `weights/adapter-manifest.json`. It refuses to
overwrite any of its output files, while allowing a base model already in
`weights/`. Transfer the Q4_0 base, seven GGUF adapters, and classifier to the
Jetson checkout's `weights/` directory; the HF snapshot and training outputs
stay on the training host.

The original checkpoint and deployment fingerprints are in
[configs/provenance.json](configs/provenance.json). They identify the
artifacts used in the experiment without redistributing them. The conversion
wrapper is new release plumbing based on the archived export procedure.
The task trainer, evaluator, and cross-profiler add a `--model` option for a
local pinned snapshot. Their training, generation, and scoring logic is
unchanged. Other included training sources retain their recorded logic.
