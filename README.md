# EdgeLoRA Bottleneck Mitigation

Research code for **Mitigating Bottlenecks in Batched Multi-Tenant LoRA Serving on Edge Devices**.

Built on [EdgeLoRA](https://github.com/shenzheyu/EdgeLoRA), this project studies two sources of delay when several requests share a base model on a Jetson: preparing an adapter that is not resident in memory, and copying the output buffer while accumulating LoRA outputs.

The repository contains the runtime changes, experiment code, and RTX training/export tools. The upstream runtime is fetched at a fixed commit and patched locally; model weights and datasets are obtained separately.

## What changes?

| Variant | Adapter preparation | LoRA output accumulation |
|---|---|---|
| **Baseline** | Synchronous | Original out-of-place accumulation |
| **Async** | Background preparation; ready requests can continue | Original accumulation |
| **In-place** | Synchronous | Update the affected columns in place when eligible |
| **Combined** | Background preparation | In-place accumulation when eligible |

Async includes cache-space reservation and owner-thread publication of prepared adapters. In-place retains a fallback for unsupported shapes. These methods address different costs; neither is expected to improve every workload.

All four variants share the experimental runtime's supporting cache, router, and output-row fixes. Baseline disables the two studied methods; it is not an unmodified upstream checkout.

**Paper scope.** The serving experiments use explicit adapter IDs to isolate runtime behavior. The router is trained and integrated, but routing quality is not the variable measured in these serving comparisons. Throughput is completed requests divided by the finite trace's elapsed time, including its drain period.

## Start here

| I want to… | Read |
|---|---|
| Inspect the C++/CUDA changes | [Runtime and patch provenance](docs/runtime.md) |
| Build and serve on Jetson | Quick start below |
| Run the paper workloads and inspect metrics | [Experiments](docs/experiments.md) |
| Train adapters/router or export GGUF artifacts | [Training and model preparation](training/README.md) |
| Read a short Korean guide | [한국어 안내](docs/README.ko.md) |

```text
patches/       Changes against the pinned official EdgeLoRA source
scripts/       Runtime setup, build, and server launcher
configs/       Portable example settings
experiments/   Serving workloads, analysis, and diagnostic code
training/      Task adapters, router, and GGUF export
docs/          Experiment mapping, provenance, and limitations
licenses/      Preserved third-party license texts
```

## Hardware and model

| Role | Recorded setup |
|---|---|
| Serving and paper measurements | NVIDIA Jetson Orin Nano Super, 8 GB unified memory, NVMe storage |
| Router and task-adapter training | NVIDIA GeForce RTX 2080 Ti |
| Base model | `meta-llama/Llama-3.2-3B-Instruct`, Q4_0 for Jetson |
| Task adapters | Six task labels, rank 16, F32 GGUF |
| Main serving configuration | 2 parallel slots, 2 task-adapter cache entries, context 2048, batch/microbatch 512 |

The base model and training datasets came from Hugging Face. The six deployed task adapters and router were trained on the RTX machine and then exported for Jetson. Their source identifiers and recorded hashes are documented under [`training/`](training/README.md). Task names such as `gpqa`, `bbh`, and `mmlupro` are runtime labels; they do not by themselves identify the training dataset.

## Quick start

Use a Jetson with its CUDA development environment, a C++ compiler, Make, Git, and Python 3.10 or newer. Keep one GPU server running at a time on the 8 GB device.

### 1. Fetch and build the runtime

```bash
git clone https://github.com/sco-edge/EdgeLoRA-Bottleneck-Mitigation.git
cd EdgeLoRA-Bottleneck-Mitigation
python3 scripts/setup_runtime.py
bash scripts/build_runtime.sh
```

The setup uses official EdgeLoRA commit `b4014595598a998db345d16cc9d3407a98998c3c` and the included patch. Build details and checks are in [docs/runtime.md](docs/runtime.md).

### 2. Prepare local model files

Follow [training/README.md](training/README.md) to prepare or train the models and export them. Place the resulting files under `weights/`, or edit the example paths:

```text
weights/
  base-q4_0.gguf
  securecode.gguf  math.gguf  ifeval.gguf
  bbh.gguf         gpqa.gguf  mmlupro.gguf
  router.gguf      router-classifier.bin
```

```bash
cp configs/runtime.example.json configs/runtime.json
python3 scripts/serve.py --config configs/runtime.json --mode combined --dry-run
```

Weights, training data, and processed splits are not included. Regenerated checkpoints need not be bit-identical to the paper artifacts. The recorded model hashes identify the artifacts used in the original measurements.

### 3. Start one variant

```bash
python3 scripts/serve.py --config configs/runtime.json --mode combined
```

Use `baseline`, `async`, `inplace`, or `combined` for `--mode`. Restart the server when changing variants. In-place is selected at server startup; asynchronous loading is selected **per request**, so the client must send the matching policy:

| Server mode | Request `adapter_load_policy` |
|---|---|
| `baseline`, `inplace` | `sync` |
| `async`, `combined` | `backend-async` |

For example, in another terminal:

```bash
curl http://127.0.0.1:8080/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain low-rank adaptation briefly.","adapter_id":1,"n_predict":32,"temperature":0,"seed":42,"adapter_load_policy":"backend-async","adapter_load_wait_timeout_ms":120000}'
```

This is an API smoke example. Use the controlled workloads in [Experiments](docs/experiments.md) for comparisons.

## Reproduce and interpret the experiments

The [experiment guide](docs/experiments.md) maps the manuscript's figures and tables to workloads, preserved source, and analysis. It also distinguishes the original measurements from release-time portability changes.

The main throughput study compares four variants at offered rates of 0.3, 0.6, and 0.9 requests/s, with 24 requests per run and three repetitions. It mixes 16-, 128-, and 256-token inputs and generates 32 output tokens. Adapter file pages are warmed; a non-resident runtime adapter does **not** mean a cold OS page cache.

The manuscript reports up to **88.9% higher median throughput** and **83.6% lower median run-mean request completion time** for Combined versus Baseline under the measured conditions. These are finite-workload results on the specified Jetson, not general capacity guarantees. See the experiment guide for TTFT, Max TBT, aggregation, and correctness limitations.

Release checks cover source provenance, patch application, script syntax, and CPU-side tooling. A fresh end-to-end CUDA build, retraining, and all GPU measurements have not been rerun as part of packaging this release.

Performance measurements do not establish output equivalence or task accuracy. Concurrent output differences remain in the original experiment records; see the [correctness limitations](docs/experiments.md) before interpreting the speed measurements.

## Attribution and license

This is an independent research extension of [EdgeLoRA](https://github.com/shenzheyu/EdgeLoRA), whose runtime derives from [llama.cpp](https://github.com/ggml-org/llama.cpp) and ggml. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md). Model and dataset terms remain separate from the code license.

The organization of this release follows research artifacts such as [Sarathi-Serve](https://github.com/microsoft/sarathi-serve), [S-LoRA](https://github.com/S-LoRA/S-LoRA), and [Punica](https://github.com/punica-ai/punica): identify the upstream system, separate setup from experiments, and document what is needed to reproduce results.

For the underlying EdgeLoRA system, please cite [**EdgeLoRA: An Efficient Multi-Tenant LLM Serving System on Edge Devices**](https://doi.org/10.1145/3711875.3729141) (ACM MobiSys 2025). The associated manuscript for this repository is **Mitigating Bottlenecks in Batched Multi-Tenant LoRA Serving on Edge Devices**. Final publication metadata and a public paper link will be added when available.
