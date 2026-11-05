# Troubleshooting

## Runtime preparation and build

| Symptom | Check | Next step |
| --- | --- | --- |
| Setup rejects the destination | Run `git -C vendor/EdgeLoRA rev-parse HEAD` | Preserve the checkout and choose a fresh `--destination`; do not reset unknown edits |
| Patch checksum mismatch | Compare `patches/runtime.patch` with the committed file | Restore the intended release through Git; do not update the manifest to hide an unexplained mismatch |
| `nvcc` missing | Run `command -v nvcc` on the Jetson | Activate the installed CUDA toolkit before building |
| Build process is killed | Inspect available memory and other GPU processes | Keep the default serial build and stop only processes you own |
| Unsupported CUDA architecture | Inspect the GPU and compiler capabilities | Use the architecture for that device; `sm_87` is the recorded Jetson default |

A custom setup destination also needs `EDGELORA_RUNTIME` for the build script
and the matching `binary` path in your local runtime JSON. These settings do
not automatically update each other. See [runtime source and build](runtime.md).

## Server launch

`Cannot load runtime config` identifies the config path and parsing/validation
error. Copy the example first; JSON does not accept comments or trailing commas.
Use [dry-run](configuration.md) to inspect paths before loading large weights.

`Missing runtime/model artifacts` lists each absent file. Verify the base,
all six task adapters, router GGUF and classifier. An export manifest does
not replace any of those files.

`Port is unavailable` can indicate another server or an invalid bind address.
Stop your previous server or select an unused port and set the same client URL.
Do not terminate unrelated processes. `Cannot execute server` usually requires
checking executable permission, target architecture and dynamic libraries.

If Async behaves like Baseline, inspect the request's `adapter_load_policy`.
Server startup alone cannot select asynchronous loading for an incoming request.
Restart the server to change in-place mode and keep client/server modes aligned.

## Benchmark failures

| Error | What to inspect |
| --- | --- |
| Linux prerequisite failure | Use dry-run on Windows; execute live cache probes on the Linux serving host |
| Missing local adapter/log | Client paths refer to its local filesystem, even when `--url` is remote |
| Slot/cache mismatch | Match `parallel_slots` and `adapter_cache_size` to the selected case |
| Dispatch lateness gate | Inspect client scheduling load; preserve the failed run rather than relaxing the 50 ms threshold |
| Missing terminal response | Check HTTP/server errors and the retained raw responses |
| Missing server token trace | Confirm the patched binary and the exact log of this server session |
| Cold-file eviction failure | Check the measured file's page residency and loader probe; do not relabel a warm run as cold |

Run into a new output directory after correcting the cause. Keep the failed
directory for diagnosis. A retry with the same block is a replacement attempt,
not an additional independent repetition for aggregation.
