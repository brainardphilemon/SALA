---
title: README.md

---

# Training a Generalist Hallucination Detector across Multiple Domains via Adaptive Layer Aggregation

[![Venue](https://img.shields.io/badge/Venue-KDD%202026-blue.svg)](https://kdd.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Official Implementation for our KDD 2026 Paper**

This repository provides the official PyTorch implementation of **SALA (Stability-Aware Adaptive Layer Aggregation)**. 

We tackle the challenge of **Multi-Domain Generalizable Hallucination Detection (MGHD)** in Large Language Models (LLMs). By mitigating the *layer-index shift* phenomenon through invariant subspace projection and adaptive layer aggregation, our method achieves robust generalization across diverse unseen target domains without requiring any test-domain supervision.
<!-- 
<p align="center">
  <img src="assets/figures/pipeline.png" alt="SALA Pipeline Overview" width="100%">
</p> -->


## 📂 Repository Structure

The codebase has been refactored and modularized for clarity:

```text
Hallucination-Detection/
├── configs/
│   └── mappings.py        # Generalization experiment settings & domain mappings
├── models/
│   ├── isr.py             # Core Invariant Subspace Risk minimization (ISR) module
│   ├── projections.py     # Layer-wise projection training and initialization
│   └── classifiers.py     # MLP classifiers
├── utils/
│   └── helpers.py         # Data processing, metric evaluation (AUC), and random seeds
├── main.py                # Main experiment pipeline
└── README.md
```


## ⚙️ Environment Setup

We recommend using a virtual environment (e.g., Conda) to run the code.

**Hardware Requirements:** * Experiments can be efficiently reproduced on a single GPU (e.g., NVIDIA A40/A100) with at least 24GB VRAM for feature projection and MLP training.

**Core Dependencies:**

* Python >= 3.8
* PyTorch >= 2.0 (with corresponding CUDA support)
* `numpy`, `scikit-learn`, `tqdm`

```bash
# Clone the repository
git clone [https://github.com/yourusername/Hallucination-Detection.git](https://github.com/yourusername/Hallucination-Detection.git)
cd Hallucination-Detection

# Install basic requirements
pip install torch torchvision torchaudio
pip install numpy scikit-learn tqdm

```

*(Note: Exact environment specifications and a `requirements.txt` will be updated shortly.)*

## 📊 Data Preparation (Feature Extraction)

Due to the massive storage footprint of layer-wise LLM hidden states (often hundreds of gigabytes per dataset), we do not host the raw `.npy` feature files or label arrays in this repository.

Users are required to extract the features and generate the hallucination labels themselves before running the ISR pipeline. Our data collection strictly follows the methodology proposed in prior works.

**Steps to generate the required data:**

1. Perform inference on your target model (e.g., `Meta-Llama-3.1-8B-Instruct`) and extract the layer-wise hidden states across the datasets (e.g., SciQ, TriviaQA, TruthfulQA).
2. Generate the corresponding hallucination labels (e.g., BLEURT scores or GPT-4 evaluated labels).
3. Save the extracted embeddings and labels as `.npy` arrays and organize them strictly into the `./data` folder matching the structure below:

```text
data/
├── save_for_eval/
│   └── [domain]_hal_det/
│       └── most_likely_[model_name]_gene_embeddings_layer_wise.npy
└── ml_[domain]_[model_name]_bleurt_score.npy

```

## 🚀 Quick Start

Once your data is prepared, you can run the main pipeline. The script handles LODO (Leave-One-Domain-Out) dimension selection, layer-wise projection, and final ERM evaluation automatically.

```bash
python main.py \
    --data_dir ./data \
    --model_name Llama3.1-8B-Instruct \
    --gengeneralize_exp G5 \
    --rep hidden \
    --use_lodo_dim

```

### Key Arguments:

* `--gengeneralize_exp`: The specific generalization mapping to run (e.g., `G1`, `G5`, `G13`). See `configs/mappings.py` for all available cross-domain setups.
* `--use_lodo_dim`: Enable source-domain LODO cross-validation for dynamic projection dimension selection.

## 📖 Citation

If you find our work or this code useful in your research, please consider citing our KDD 2026 paper:

```bibtex
@inproceedings{yourname2026sala,
  title={Training a Generalist Hallucination Detector across Multiple Domains via Adaptive Layer Aggregation},
  author={[Your Name]},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2026}
}

```

```

```