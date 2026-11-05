# Which hashes describe which artifact?

| Record | Identifies | Does not establish |
| --- | --- | --- |
| Git commit | This public source tree and its documentation | The state of an unrecorded device checkout |
| Runtime manifest | Pinned upstream, patch bytes, reconstructed runtime files | A completed CUDA build |
| Experiment source provenance | Original archived scripts behind the portable extraction | Byte identity of every public wrapper |
| Training provenance | Archived source/checkpoints and deployed artifact hashes | Identical weights after retraining |
| Export manifest | Input and output hashes for one new conversion | Historical checkpoint identity unless hashes match |

The router rename mapping in [training instructions](../training/README.md)
connects current entry points to historical filenames. Keep the original keys
in provenance records; renaming an entry point does not rewrite the archive.

When adding new measurements, record the public commit plus any uncommitted
diff, the local model hashes and the environment. Use a new result directory.
If a source patch changes, create new provenance for that revision rather than
replacing historical checksums without retaining the old source relationship.
