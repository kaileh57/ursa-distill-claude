#!/bin/bash
# Complete platform setup script

set -e

echo "🚀 Setting up Ursa Minor 7B Distillation Platform"

# 1. Environment setup
echo "📦 Setting up environment..."
chmod +x setup_environment.sh
./setup_environment.sh

# 2. Data preparation
echo "📊 Preparing dataset..."
python data/prepare_dataset.py \
    --input_dataset "qfq/s1K" \
    --output_dataset "ursa-minor/s1K-7b-tokenized" \
    --num_proc 8

# 3. Setup evaluation environment
echo "🔍 Setting up evaluation..."
chmod +x eval/setup_evaluation.sh
./eval/setup_evaluation.sh

# 4. Create necessary directories
echo "📁 Creating directories..."
mkdir -p {checkpoints,logs,results/{training,evaluation,inference}}

# 5. Make scripts executable
echo "🔧 Setting permissions..."
chmod +x train/sft_7b.sh
chmod +x eval/evaluate_7b.py
chmod +x inference/inference_7b.py
chmod +x scripts/monitor_training.py

echo "✅ Platform setup complete!"
echo ""
echo "Next steps:"
echo "1. Update wandb entity in configs"
echo "2. Run training: bash train/sft_7b.sh"
echo "3. Monitor training: python scripts/monitor_training.py --run_name YOUR_RUN_NAME"
echo "4. Evaluate model: python eval/evaluate_7b.py --model_path PATH_TO_MODEL"