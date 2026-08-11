# Thesis: Molecular Dynamics Trajectory Embeddings

Short README for the repository used in the Master's thesis project on learning embeddings for molecular dynamics trajectories.

**Contents:**
- **Description:** high-level project goals and approach
- **Getting started:** environment and dependencies
- **Usage:** data, training, and experiments
- **Project structure:** key folders and files
- **Notes:** tips for common tasks

**Description**:
- **Goal:** Build and evaluate trajectory-level embeddings for molecular dynamics data to support downstream tasks (classification, clustering, interpretability).
- **Approach:** Preprocess molecular trajectory data, train models (CNNs and other architectures) to produce embeddings, and analyze downstream performance and interpretability.

**Getting started**:
- Python: recommended 3.11/3.12 (project contains a `gpulab_env` virtualenv configured for Python 3.12).
- Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- If you prefer the interpretability environment, use `requirements-interpretability.txt`.

**Running experiments**:
- A convenience script `run_experiment.sh` exists for running typical experiments. Many experiments are implemented under `src/experiments/`.
- Example: run the simple CNN experiment (uses the workspace venv or Docker task):

```bash
python src/experiments/SCNN.py
```

**Data**:
- Cached and preprocessed datasets live under `hf_cache/` and mapped Arrow files under `hf_cache/mapped_cache/`.
- Preprocessing code is in `src/data_preprocessing/` and `src/data_loading/`.

**Project structure (high level)**:
- `src/` : source code
  - `data_preprocessing/` : preprocessing pipelines (e.g., `pre_processing_huggingface.py`)
  - `data_loading/` : dataset loaders
  - `experiments/` : experiment entry points and scripts
  - `model_training/`, `Models/` : model implementations and training utilities
  - `utils/` : helper utilities and common functions
  - `interpretability/`: interpretability scripts
  - `models/`: model definitions and architectures 
- `hf_cache/` : local cache of datasets / transformed Arrow shards
- `jupyter_notebooks/` : exploratory analysis and debugging notebooks
- `output/` : trained models, plots, and profiling outputs
- `CLI/` : scripts for running command-line experiments 
