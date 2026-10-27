# Runtime source and build

The runtime is released as a patch against [EdgeLoRA](https://github.com/shenzheyu/EdgeLoRA)
commit `b4014595598a998db345d16cc9d3407a98998c3c`. Setup reconstructs the
experiment runtime under `vendor/EdgeLoRA/edgelora`; the upstream checkout is not
included in this repository.

```bash
python3 scripts/setup_runtime.py
bash scripts/build_runtime.sh
```

The build requires a Linux Jetson Orin environment with GNU make, GCC/G++, and
the JetPack CUDA toolkit (`nvcc` on `PATH`). It uses the original GNU make build
and produces `vendor/EdgeLoRA/edgelora/llama-server` and `llama-quantize`.
Compilation uses one job and CUDA architecture `sm_87` by default. The script
does not install or change the device's CUDA toolkit.

For a different build location or architecture, set `EDGELORA_RUNTIME`,
`CUDA_ARCH`, and optionally `BUILD_JOBS`. The paper's serving experiments use
Jetson Orin; an RTX build is not a reproduction of those performance results.
Use one serving process at a time on the 8 GB device.

## What the patch contains

| Files within `edgelora/` | Purpose |
|---|---|
| `examples/server/server.cpp` | Background adapter-loading worker, loading state transitions, safe publication, and router integration |
| `src/llama.cpp`, `include/llama.h` | Cache reservation/materialization/publication, adapter lifetime handling, and LoRA graph construction |
| `src/llama-lora-output-map.h` | Keep adapter IDs aligned with selected output rows |
| `ggml/src/ggml-cuda/acc.cu` | Checked in-place column accumulation with the generic operation retained as fallback |
| `common/common.*`, `common/router-classifier.*` | Adapter registration, resident router, classifier scoring, and runtime options |
| `convert_lora_to_gguf.py` | Conversion support used by the deployed artifacts |
| `Makefile`, `common/CMakeLists.txt`, `tests/*` | Build integration and the existing router regression test |

The patch includes supporting cache, routing, and output-row fixes as well as
the two methods studied in the paper. **Baseline** in the experiments means
the repaired experimental runtime with both methods disabled. It does not
mean an unmodified checkout of the original EdgeLoRA release.

## Select a method

In-place accumulation is selected when the server process starts. Adapter
loading policy is selected separately in each `/completion` request.

| Method | Server environment `EDGELORA_ACC_AUTO` | Request field `adapter_load_policy` |
|---|---|---|
| Baseline | `0` | `"sync"` |
| Async | `0` | `"backend-async"` |
| In-place | `1` | `"sync"` |
| Combined | `1` | `"backend-async"` |

An omitted loading policy defaults to `sync`. Async is therefore not enabled
by setting an environment variable alone. Restart the server when changing
`EDGELORA_ACC_AUTO`; it is cached in the process.

Leave `EDGELORA_ACC_SLICE`, `EDGELORA_ACC_AUTO_FORCE_FALLBACK`, and
`EDGELORA_ACC_AUTO_TRACE` unset for the paper configurations. These switches
are diagnostic paths retained in the original experiment code.

The in-place graph rewrite checks shape, ownership, and liveness before
reusing an output buffer. The CUDA dispatcher also checks the relevant tensor
layout and aliasing conditions. Unsupported cases keep the generic operation.
Async prepares an adapter in a worker and publishes it from the serving thread
at a safe point; it does not imply that GPU transfer and computation always
overlap.

## Adapter and router interface

Use `--batch-lora true --lora-init-without-apply` and one `--lora FILE` per
artifact. Their command-line order defines adapter IDs. The recorded six-task
mapping is `securecode=0`, `math=1`, `ifeval=2`, `bbh=3`, `gpqa=4`, and
`mmlupro=5`, followed by the resident router at ID `6`.

The router options are:

```text
--router-classifier PATH/TO/router-classifier.bin
--router-lora-id 6
--router-label-map 0,1,2,3,4,5
--router-top-k 1
```

The router remains resident and is excluded from the task-adapter cache.
`--adapter-cache-size` controls that task cache. The runtime preloads some task
adapters at startup, so the benchmark scripts establish the desired resident
or non-resident condition before measuring.

Requests with an explicit `adapter_id` bypass routing. To measure routing,
omit `adapter_id` and provide raw `router_text` when required by the request
format. `adapter_load_wait_timeout_ms` defaults to 60000. Do not enable the
failure-injection field in normal runs.

## Source verification and scope

`patches/runtime_manifest.json` records the upstream commit, patch checksum,
and SHA-256 of each of the 14 changed or added runtime files. Those files were
copied from the final `lora-acc-auto-20260929-v1` source snapshot and checked
against hashes computed on the source device. Generated web headers,
`build-info.cpp`, object files, executables, model weights, and machine-local
files are excluded; the build regenerates its required generated files.

```bash
python3 scripts/setup_runtime.py --check
```

Setup refuses to overwrite an existing destination that does not match the
pinned, patched runtime. Choose a fresh `--destination` if a previous setup was
interrupted. The release was checked by applying the patch to a clean pinned
checkout and comparing source hashes. Compilation and GPU experiments were
not rerun while packaging the release.
