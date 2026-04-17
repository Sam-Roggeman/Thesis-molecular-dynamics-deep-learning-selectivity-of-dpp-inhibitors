#!/bin/bash
cd /project_antwerp/Thesis-molecular_dynamics_trajectory_embeddings || exit

token=$(cat '/project_antwerp/token.txt')
echo "machine github.com login oauth2 password ${token}" >> ~/.netrc
git pull

# take the first argument as the filename of the python script to run
PYTHON_MODULE=$1
python3 -m "$PYTHON_MODULE"
