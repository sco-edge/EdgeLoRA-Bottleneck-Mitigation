# Export inputs and recovery

Complete [conversion environment setup](README.md) before exporting. The
wrapper expects `adapter_config.json` and `adapter_model.safetensors` in each
of the six task-named folders and in the router checkpoint folder. It does
not accept a directory of already converted GGUF files as PEFT inputs.

The wrapper runs the patched converter on CPU with Hugging Face and
Transformers offline flags. The local base snapshot must already contain
required model/tokenizer files. Network access during conversion is not a
substitute for completing the download stage.

Before conversion it hashes all checkpoint inputs and refuses output filename
collisions. After conversion it compares the classifier payload with the saved
PEFT score tensor, verifies the 6-by-3072 contract and rechecks input hashes.

If conversion fails halfway, retain the partial folder and logs. Retry with a
new `--output-dir`; the wrapper will not overwrite partially generated files.
`adapter-manifest.json` is written only after all conversion checks succeed.
The base F16/Q4_0 conversion is a separate step and is not covered by this
adapter export manifest.

## Verify files after transfer

Copy the eight exported files and `adapter-manifest.json` together. On the
destination host, from the repository root:

```bash
python scripts/verify_artifacts.py weights
```

The verifier streams each named file through SHA-256 and exits nonzero for
missing files, corruption, invalid digests or unsafe manifest filenames. It
does not load models or require CUDA. It checks only files named in `outputs`;
unrelated files in the directory are not included in its success statement.

For a manifest stored elsewhere, pass `--manifest PATH`. Compare the manifest
itself with the source copy through a trusted transfer. Checksums detect
changes relative to a manifest; they do not authenticate who produced it.
Verify the base model separately because the adapter manifest excludes it.

`source_inputs` records checkpoint hashes; `outputs` records exported file
hashes. `classifier_bit_exact` describes the export-time comparison with the
PEFT classifier, not equivalence between serving outputs across configurations.
