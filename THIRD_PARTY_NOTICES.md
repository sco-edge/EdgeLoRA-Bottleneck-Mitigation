# Third-party notices

## EdgeLoRA

- Source: https://github.com/shenzheyu/EdgeLoRA
- Official base commit: `b4014595598a998db345d16cc9d3407a98998c3c`
- Paper: [*EdgeLoRA: An Efficient Multi-Tenant LLM Serving System on Edge Devices*](https://doi.org/10.1145/3711875.3729141), ACM MobiSys 2025.
- The upstream [artifact appendix](https://github.com/shenzheyu/EdgeLoRA/blob/b4014595598a998db345d16cc9d3407a98998c3c/268_artifact.pdf), section A.2, identifies the artifact code license as MIT.

This release supplies research changes to EdgeLoRA and related training and evaluation tools. The runtime is reconstructed from the official commit and the included patch. The patch includes prerequisite integration and correctness work as well as the two techniques described in the manuscript; it should not be interpreted as a two-function-only diff. Detailed file fingerprints are in `patches/runtime_manifest.json`.

## llama.cpp, ggml, and GGUF

EdgeLoRA's `edgelora/` runtime derives from llama.cpp and ggml. The original MIT notice, including `Copyright (c) 2023-2024 The ggml authors`, is preserved in [licenses/EdgeLoRA-ggml-MIT.txt](licenses/EdgeLoRA-ggml-MIT.txt). The GGUF Python package notice, `Copyright (c) 2023 Georgi Gerganov`, is preserved in [licenses/GGUF-py-MIT.txt](licenses/GGUF-py-MIT.txt).

Setup downloads the original source tree, including its remaining component-specific notices. Those notices continue to apply. Dependencies installed separately retain their own licenses.

## Models and datasets

The code license does not license Meta Llama model weights, third-party datasets, or their contents. Obtain those resources from their original publishers under the applicable terms. This repository contains no model weights, training datasets, or generated-answer corpora. Dataset identifiers, known revisions, and artifact fingerprints are listed in [training/README.md](training/README.md) and `training/configs/`.
