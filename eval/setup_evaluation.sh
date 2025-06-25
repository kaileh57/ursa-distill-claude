#!/bin/bash
# Setup evaluation environment

# Clone and setup lm-evaluation-harness
cd eval
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout 4cec66e4e468d15789473d6d63c3a61a751fa524

# Install evaluation dependencies
pip install -e .[math,vllm]

echo "Evaluation environment setup complete"