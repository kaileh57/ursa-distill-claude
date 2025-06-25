# Ursa Minor 7B: Claude Distillation System

A complete distillation system for training Qwen2.5-7B to behave like Claude using the S1K research paper techniques and dataset.

## Quick Start

### Single Command Execution
```bash
# Run the complete pipeline
python run_distillation.py
```

### Manual Step-by-Step
```bash
# 1. Setup platform
bash setup_complete_platform.sh

# 2. Start training  
bash train/sft_7b.sh

# 3. Monitor training (in another terminal)
python scripts/monitor_training.py --run_name ursa-minor-7b-TIMESTAMP

# 4. Evaluate model
python eval/evaluate_7b.py --model_path checkpoints/ursa-minor-7b-TIMESTAMP

# 5. Test inference
python inference/inference_7b.py \
    --model_path checkpoints/ursa-minor-7b-TIMESTAMP \
    --prompt "How many r's are in the word 'raspberry'?"
```

## System Requirements

### Hardware
- **GPU**: RTX A6000 + RTX A4500 (or equivalent with 40GB+ VRAM total)
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ free space
- **Training Time**: ~8 hours max

### Software
- Python 3.10+
- CUDA 11.8+
- Conda/Mamba
- Git

## Configuration

Before running, update these settings:

1. **Wandb Entity**: Edit `train/sft_7b.py` line 254
2. **HuggingFace Organization**: Update dataset paths if using private datasets
3. **Model Hub**: Update `hub_model_id` in training script

## Pipeline Components

### 1. Data Pipeline (`data/`)
- Downloads and tokenizes S1K dataset
- Formats for Qwen2.5 chat template
- Handles thinking trajectories

### 2. Training (`train/`)
- FSDP-optimized training for 7B model
- Flash Attention 2 support
- Wandb integration for monitoring

### 3. Evaluation (`eval/`)
- LM Evaluation Harness integration
- AIME, OpenAI Math, GPQA benchmarks
- Custom evaluation metrics

### 4. Inference (`inference/`)
- vLLM-powered fast inference
- Batch processing support
- Interactive testing

### 5. Monitoring (`scripts/`)
- Real-time training monitoring
- Metric visualization
- Progress tracking

## Command Line Options

### Main Pipeline
```bash
python run_distillation.py [OPTIONS]

Options:
  --skip-setup          Skip environment setup
  --skip-data           Skip data preparation  
  --skip-training       Skip training phase
  --model-path PATH     Use existing model for eval/inference
  --eval-only           Only run evaluation on existing model
```

### Individual Components
```bash
# Data preparation
python data/prepare_dataset.py --input_dataset qfq/s1K --num_proc 8

# Training
bash train/sft_7b.sh

# Evaluation  
python eval/evaluate_7b.py --model_path PATH --tasks "aime24,openai_math"

# Inference
python inference/inference_7b.py --model_path PATH --prompt "Your question"

# Monitoring
python scripts/monitor_training.py --run_name RUN_NAME
```

## Expected Results

### Training Metrics
- **Loss**: Should decrease from ~2.0 to ~0.5
- **Learning Rate**: Cosine schedule from 2e-5 to 0
- **Training Time**: 6-8 hours on dual GPU setup

### Evaluation Benchmarks
- **AIME**: Target 15-25% accuracy
- **OpenAI Math**: Target 40-60% accuracy  
- **GPQA**: Target 35-50% accuracy

### Model Behavior
- Step-by-step reasoning in responses
- Claude-like helpful, harmless, honest behavior
- Improved mathematical problem-solving

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce `micro_batch_size` in training script
- Enable `gradient_checkpointing=True`
- Use `bf16=True` instead of fp32

**Dataset Loading Errors**  
- Check internet connection for HF Hub access
- Verify dataset permissions
- Try reducing `num_proc` in data preparation

**Training Crashes**
- Check GPU memory with `nvidia-smi`
- Verify FSDP configuration
- Review wandb logs for specific errors

**Evaluation Failures**
- Ensure evaluation environment is set up: `bash eval/setup_evaluation.sh`
- Check model path exists and is accessible
- Verify vLLM compatibility with model

### Performance Optimization
- Use `tensor_parallel_size=2` for dual GPU inference
- Enable `flash_attention_2` for training speedup
- Adjust `gradient_accumulation_steps` based on available memory

## File Structure
```
ursa-distill-claude/
├── data/                    # Data processing pipeline
├── train/                   # Training scripts and configs
├── eval/                    # Evaluation pipeline
├── inference/               # Inference utilities
├── scripts/                 # Monitoring and utilities
├── configs/                 # Training configurations
├── checkpoints/             # Model checkpoints (created during training)
├── results/                 # Evaluation and inference results
├── run_distillation.py      # Main execution script
├── setup_complete_platform.sh  # Complete setup script
└── README.md               # This file
```

## License

This project is for research and educational purposes. Please respect the licenses of the underlying models and datasets:
- Qwen2.5: Apache 2.0
- S1K Dataset: Check dataset license
- Evaluation datasets: Various (see lm-evaluation-harness)