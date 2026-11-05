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
