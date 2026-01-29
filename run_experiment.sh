#!/bin/bash
# cd to the root directory of the project
cd /project_antwerp/Thesis-molecular_dynamics_trajectory_embeddings || exit
export HF_HOME="/project_scratch/hf_cache"
# Update
sudo apt update
# Install the datasets library
pip install -r ./src/requirements.txt

# take the first argument as the filename of the python script to run
PYTHON_MODULE=$1
python3 -m "$PYTHON_MODULE"
