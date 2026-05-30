---
title: README.md

---

# Training a Generalist Hallucination Detector across Multiple Domains via Adaptive Layer Aggregation

[![Venue](https://img.shields.io/badge/Venue-KDD%202026-blue.svg)](https://kdd2026.kdd.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation for the KDD 2026 paper:

**Training a Generalist Hallucination Detector across Multiple Domains via Adaptive Layer Aggregation**

This repository implements **SALA** (**S**tability-**A**ware **L**ayer **A**ggregation), a method for **Multi-Domain Generalizable Hallucination Detection (MGHD)** in large language models. SALA is designed to train hallucination detectors on multiple labeled source domains and generalize to unseen target domains without requiring target-domain supervision.

The method addresses the **layer-index shift** phenomenon by learning layer-specific invariant subspace projections and adaptively aggregating stable discriminative signals across LLM layers.

<!--
<p align="center">
  <img src="assets/figures/pipeline.png" alt="SALA Pipeline Overview" width="100%">
</p>
-->

## Repository Structure

```text
Hallucination-Detection/
├── configs/
│   └── mappings.py        # Generalization settings and source/target domain mappings
├── models/
│   ├── isr.py             # ISR-style invariant subspace recovery utilities adapted to SALA
│   ├── projections.py     # Layer-wise projection training and initialization
│   └── classifiers.py     # MLP classifiers for hallucination detection
├── utils/
│   └── helpers.py         # Data loading, metric computation, and random seed utilities
├── main.py                # Main experiment pipeline
├── supplementary.md
└── README.md

```

## Environment Setup

We recommend using a virtual environment such as Conda.

### Hardware Requirements

The projection and detector-training stages can be reproduced on a single GPU. We recommend a GPU with at least 24GB VRAM, such as an NVIDIA A40, A100, or comparable device, depending on the number of layers, domains, and cached feature size.

### Core Dependencies

* Python >= 3.8
* PyTorch >= 2.0
* NumPy
* scikit-learn
* tqdm

```bash
# Clone the repository
git clone https://github.com/Nellie179/SALA.git
cd SALA

# Install basic requirements
pip install torch torchvision torchaudio
pip install numpy scikit-learn tqdm
```

A complete `requirements.txt` file will be provided with the final release.

## Data Preparation

This repository does not host raw LLM hidden-state caches or label arrays because layer-wise hidden states can require hundreds of gigabytes of storage.

Before running SALA, users need to prepare:

1. Layer-wise hidden-state embeddings extracted from the target LLM.
2. Binary hallucination labels for each response.
3. Source/target domain mappings following the experiment configuration.

The expected data structure is:

```text
data/
├── save_for_eval/
│   └── [domain]_hal_det/
│       └── most_likely_[model_name]_gene_embeddings_layer_wise.npy
└── ml_[domain]_[model_name]_bleurt_score.npy
```

Here, `[domain]` denotes the dataset/domain name, and `[model_name]` denotes the evaluated LLM backbone.

## Quick Start

After preparing the required feature and label files, run:

```bash
python main.py \
    --data_dir ./data \
    --model_name Llama3.1-8B-Instruct \
    --gengeneralize_exp G5 \
    --rep hidden \
    --use_lodo_dim
```

### Key Arguments

* `--data_dir`: Path to the prepared feature and label files.
* `--model_name`: Name of the LLM backbone used for feature extraction.
* `--gengeneralize_exp`: Cross-domain generalization setting, such as `G1`, `G5`, or `G13`. See `configs/mappings.py` for all supported settings.
* `--rep`: Representation type used for detector training.
* `--use_lodo_dim`: Enables leave-one-domain-out source validation for projection-dimension selection.

## Acknowledgements

The ISR-style invariant subspace recovery component in `models/isr.py` is adapted from the official implementation of **Invariant-feature Subspace Recovery (ISR)**:

https://github.com/uiuctml/ISR

Original paper:

Haoxiang Wang, Haozhe Si, Bo Li, and Han Zhao.
"Provable Domain Generalization via Invariant-Feature Subspace Recovery."
International Conference on Machine Learning (ICML), 2022.

The original ISR implementation is released under the MIT License. We adapt the ISR procedure to the multi-domain hallucination detection setting and integrate it into our feature-cache and evaluation pipeline.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{li2026sala,
  title     = {Training a Generalist Hallucination Detector across Multiple Domains via Adaptive Layer Aggregation},
  author    = {Li, Xinyi and Deng, Yongxin and others},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3770855.3817792}
}
```

The official ACM Digital Library BibTeX entry will be added once it becomes available.

## License

This project is released under the MIT License. See the `LICENSE` file for details.

Portions of the ISR-style baseline implementation are adapted from the official ISR codebase, which is also released under the MIT License.
