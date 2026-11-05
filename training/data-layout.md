# Prepared task-data layout

`train_task_adapter.py --data-root ROOT --task math` reads
`ROOT/math/train.jsonl` and `ROOT/math/validation.jsonl`. Cross-profiling reads
the selected split, commonly `test.jsonl`, from each task's supplied data root.
Raw download directories and these prepared JSONL files are different formats.

Each training row has a `messages` list in exactly this order:

```json
{"messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"4"}]}
```

This synthetic row illustrates the schema; it is not a paper training example.
All message content must be nonempty. The trainer applies the base tokenizer's
chat template and masks tokens before the assistant answer. It verifies that
the prompt prefix matches the complete chat serialization.

At the fixed 512-token cutoff, an entirely truncated assistant answer is an
error. Do not disable token validation to make a dataset appear compatible.
Record the preparation script, source revisions and resulting split hashes
alongside training outputs. See [training setup](README.md) for actual sources.
