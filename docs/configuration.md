# Runtime configuration reference

Start with [runtime.example.json](../configs/runtime.example.json). The launcher
resolves paths **inside the JSON** relative to the repository root, regardless
of the current working directory. The `--config` filename itself follows the
shell's current directory. Absolute paths and `~` in artifact paths are supported.

| Key | Default/example | Meaning |
| --- | --- | --- |
| `binary` | `vendor/EdgeLoRA/edgelora/llama-server` | Patched executable |
| `base_model` | `weights/base-q4_0.gguf` | Quantized base model |
| `task_adapters` | Six paths | Ordered IDs 0..5; do not alphabetize |
| `router_adapter` | `weights/router.gguf` | Appended as adapter ID 6 |
| `router_classifier` | `weights/router-classifier.bin` | Separate classifier |
| `slot_save_path` | `results/slots` | Writable slot-state directory |
| `context_size` | 2048 | Server context capacity |
| `batch_size`, `micro_batch_size` | 512, 512 | Logical/physical batch sizes |
| `parallel_slots` | 2 | Concurrent serving slots |
| `adapter_cache_size` | 2 | Task-adapter cache entries; excludes router |
| `threads` | 4 | CPU thread count |
| `gpu_layers` | 99 | Layer-offload argument, not a measured layer count |

The four artifact fields and six adapter paths are required. Numeric size fields
must be positive integers; `gpu_layers` may be zero. Defaults apply when optional
numeric fields are absent. The launcher validates types, not GPU memory fit.

```bash
python scripts/serve.py --config configs/runtime.json --mode combined --dry-run
```

Dry-run prints the resolved command, selected environment flags and required
request policy. It does not check that weights exist, bind a port or load a model.
