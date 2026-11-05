# Reading the cache conditions

Two different residency states matter in these experiments:

| Condition | Runtime task adapter | Adapter file pages |
| --- | --- | --- |
| Resident | Already available to the runtime | Warm in the ablation |
| Nonresident | Needs runtime preparation | Explicitly warmed |
| Cold | Needs runtime preparation | Verified absent before loader file open |

Restarting the server alone does not make file pages cold. Conversely, removing
file pages does not prove that an already materialized runtime adapter must
be loaded again. The client prepares and checks both states separately.

The cold-file method targets one adapter with `posix_fadvise` and checks page
residency with `mincore`. The preload observer checks residency again at loader
entry. Preserve both observations with a cold run. The portable release does
not include the original device-specific read-counter gate.

Host-side background preparation, GPU copy/compute overlap and file-cache
residency are separate claims. A speed difference alone does not identify which
overlaps occurred. See [cold-run commands](experiments.md) for the probe setup.
