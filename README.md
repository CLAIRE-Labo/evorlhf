# Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning

![Method Image](./data/readme/method-fig.png)

[![Paper](https://img.shields.io/badge/Paper-arXiv%20preprint-b31b1b.svg)](https://arxiv.org/abs/2504.05108)
[![License](https://img.shields.io/github/license/CLAIRE-Labo/EvoTune)](./LICENSE)

Official repository for the paper:

> **Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning**  
> Anja Surina, Amin Mansouri, Lars Quaedvlieg, Amal Seddas, Maryna Viazovska, Emmanuel Abbe, Caglar Gulcehre

### Citation
```bibtex
@article{surina2025algorithm,
  title={Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning},
  author={Surina, Anja and Mansouri, Amin and Quaedvlieg, Lars and Seddas, Amal and Viazovska, Maryna and Abbe, Emmanuel and Gulcehre, Caglar},
  journal={arXiv preprint arXiv:2504.05108},
  year={2025}
}
```

---

## 🧠 Overview

**EvoTune** is a framework for discovering new algorithms by combining:

1. 🧬 Evolutionary search over LLM-generated Python programs, and
2. 🏆 Reinforcement Learning to fine-tune the LLM based on relative program performances.

We show that this synergy improves both the performance and uniqueness of solutions across classic combinatorial problems, including:
- ✅ Bin Packing
- ✅ Traveling Salesman Problem (TSP)
- ✅ FlatPack Problem

---

## 📦 Repo Structure

The core codebase lives under ```src/``` and is organized as follows:

```plaintext
projects/
├── evotune/
│   ├── configs/                  # Hydra-based config system
│   │   ├── accelerate_config/    # Accelerate (multi-GPU) configs
│   │   ├── cluster/              # SLURM / cluster overrides
│   │   ├── model/                # Model-specific settings
│   │   ├── sweep/                # Sweep configuration files
│   │   ├── task/                 # Per-task configs (e.g., bin, tsp, etc.)
│   │   ├── train/                # Training configuration
│   │   └── config.yaml           # Default config
│   ├── data/                     # Example datasets and readme assets
│   ├── installation/             # Dockerfiles for various hardware (CUDA, TGI, vLLM)
│   ├── scripts/                  # Example launch scripts for sweeps
│   │   ├── run_eval_sweep_example.sh
│   │   └── run_train_sweep_example.sh
│   └── src/
│       ├── packing/             # Core EvoTune framework
│       │   ├── evaluate/        # Task-specific logic (registered via registry)
│       │   │   ├── bin_packing/
│       │   │   ├── flat_pack/
│       │   │   ├── tsp/
│       │   │   ├── __init__.py  # Auto-imports task modules
│       │   │   ├── registry.py  # Central task registry
│       │   │   └── README.md    # How to add new tasks
│       │   ├── funsearch/       # Program database
│       │   ├── logging/         # Logging, statistics, and function tracking
│       │   ├── model/           # Prompting, LLM I/O, inference engine setup
│       │   ├── parallel/        # Multiprocessing producers & consumers
│       │   ├── train/           # DPO pipelines for fine-tuning LLMs
│       │   └── utils/           # Seeding, function helpers, etc.
│       ├── experiments/         # Scripts for specific experiments (train / eval)
│       └── plotting/            # Utilities for generating paper plots
├── .pre-commit-config.yaml      # Code formatting & linting hooks
├── pyproject.toml               # Build & dependency configuration
├── LICENSE                      # License (MIT)
```

---

## ⚙️ Setup & Dependencies

To run EvoTune, you **must use a Docker container** due to dependencies on vLLM, JAX, and CUDA-incompatible multiprocessing. We provide optimized Dockerfiles under:

```plaintext
installation/
├── docker-amd64-cuda-tgi/   # For x86_64 machines using TGI
├── docker-amd64-cuda-vllm/  # For x86_64 machines using vLLM
├── docker-arm64-cuda/       # For Apple Silicon (ARM64 + CUDA support)
```

Or, pull directly from Docker Hub:

```bash
# For AMD64
docker pull larsquaedvlieg/evotune:amd64-cuda

# For ARM64 (will be pushed to the hub soon)
docker pull larsquaedvlieg/evotune::arm64-cuda
```

> ✅ Most experiments were run using a **single A100 GPU (40GB)**. Lower memory GPUs (24GB or lower) may require reduced configurations.

---

## 🚀 How to Run the Code

You can run EvoTune either locally (inside Docker) or on a cluster.

### ▶️ Local Runs

The two main entry points are located in:

```plaintext
src/experiments/
├── main.py   # For running training with evolution + finetuning
├── eval.py   # For evaluating saved program banks
```

### 📡 Cluster / Sweep Runs

We provide example sweep scripts in the ```scripts/``` folder:

```plaintext
scripts/
├── run_eval_sweep_example.sh
├── run_train_sweep_example.sh
```

These are designed to be used with job schedulers like SLURM or RunAI. To use them:

1. Fill in the ```# TODO``` block in each script with your cluster submission logic.
2. Configure the sweep/grid settings in the appropriate ```configs/sweep/``` and ```configs/cluster/``` YAML files.
3. Launch your sweep using the modified script.

> You can also run hyperparameter or task sweeps locally by adapting these scripts — just remove the SLURM logic.

---

## 🧱 Adding a New Task

To add your own task:

👉 Navigate to:

```src/packing/evaluate/README.md```

You’ll find clear instructions for implementing and registering:

- ```generate_input```
- ```evaluate_func```
- ```get_initial_func```
- ```system_prompt``` / ```append_prompt```

The registry-based design ensures zero boilerplate in the main training loop.

---