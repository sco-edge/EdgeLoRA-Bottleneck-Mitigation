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
