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
