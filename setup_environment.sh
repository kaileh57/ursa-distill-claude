#!/bin/bash
# setup_environment.sh

# Create conda environment
conda create -n ursa-minor python=3.10 -y
conda activate ursa-minor

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Setup wandb (requires manual login)
wandb login

echo "Environment setup complete. Remember to activate with: conda activate ursa-minor"