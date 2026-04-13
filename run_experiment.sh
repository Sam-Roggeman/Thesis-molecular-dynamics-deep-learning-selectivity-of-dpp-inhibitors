#!/bin/bash
cd /project_antwerp/Thesis-molecular_dynamics_trajectory_embeddings || exit
<<<<<<< HEAD
export HF_HOME="/project_scratch/hf_cache"
# Update
# Install the datasets library
pip install -r ./src/requirements.txt
=======
token=$(cat '/project_antwerp/token.txt')
echo "machine github.com login oauth2 password ${token}" >> ~/.netrc
git pull
>>>>>>> 166d335 (updated run experiment)

# take the first argument as the filename of the python script to run
PYTHON_MODULE=$1
python3 -m "$PYTHON_MODULE"
