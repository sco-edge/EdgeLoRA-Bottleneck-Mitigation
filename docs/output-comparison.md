# Compare generated outputs separately from performance

First match configurations by block, prompt token IDs, requested adapter,
decoding settings and output-token count. Compare request labels, not JSONL
line order: throughput workers append records when requests complete.

Throughput response records contain a SHA-256 of concatenated nonempty SSE
text. Latency trials also retain server token IDs in `token-ids.json`, after
validating task/slot traces. Compare token sequences when available and use
text hashes as an additional observation. These answer different questions.

A mismatch should retain both raw outputs, their request payloads, source/model
revisions and the corresponding logs. Report how many individual requests and
how many complete matched runs agree. Do not remove mismatched pairs solely
because they complicate a performance result.

Matching hashes do not establish task accuracy, and valid timing contracts do
not establish output equivalence. Historical concurrent output differences
remain documented in [the experiment guide](experiments.md); this maintenance
update adds no claim that their cause has been found or corrected.
