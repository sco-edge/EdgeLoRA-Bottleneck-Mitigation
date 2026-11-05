# Terms used in this repository

| Term | Meaning here |
| --- | --- |
| Task adapter | One of six LoRA adapters with fixed runtime IDs 0..5 |
| Router | Six-label classifier plus its LoRA adapter, resident at ID 6 |
| Explicit adapter ID | Request selection that bypasses router inference |
| Slot | Server request-processing slot; distinct from a cache entry |
| Cache entry | Capacity for a materialized task adapter, excluding the router |
| Block | Matched repetition index that fixes trace/arrival randomness |
| Offered rate | Scheduled arrivals per second, independent of completion |
| Drain | Time after the last scheduled arrival while requests finish |
| TTFT | Client time until the first nonempty text event |
| Max TBT | Largest consecutive server token-generation gap for a request |
| SSE event | HTTP stream event; may contain no text and is not necessarily one token |
| In-place | Eligible accumulation updates an existing output buffer |
| Async | Adapter preparation runs in a worker with serving-thread publication |
| GGUF | Runtime deployment artifact format used for base/adapters |
| PEFT checkpoint | Training output before conversion to deployment artifacts |

Baseline includes the shared supporting runtime fixes. It does not denote the
unmodified upstream release. See [runtime details](runtime.md) and
[cache conditions](cache-conditions.md) when interpreting those terms.
