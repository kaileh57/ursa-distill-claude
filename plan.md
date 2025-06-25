# Ursa Minor 7B Distillation Platform Setup Plan

## Overview
Complete setup plan for distilling Qwen-2.5-7B-Instruct to behave like Claude using s1K techniques and your existing dataset. This plan provides everything needed for a coding agent to build the complete training platform.

## 1. Project Structure Setup

```bash
# Clone base repository and setup project structure
mkdir ursa-minor-7b
cd ursa-minor-7b

# Create directory structure
mkdir -p {data,train,eval,scripts,configs,results,checkpoints}
mkdir -p data/{raw,processed,tokenized}
mkdir -p train/{logs,scripts}
mkdir -p eval/{lm-evaluation-harness,custom}
mkdir -p results/{inference,evaluation,grading}
mkdir -p configs/{training,evaluation}
```

## 2. Environment Setup

### requirements.txt
```txt
torch==2.5.1
transformers==4.46.1
trl==0.12.0
datasets==3.1.0
accelerate==1.0.1
vllm==0.6.4.post1
wandb==0.17.3
openai==1.56.1
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
tqdm>=4.62.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
ipykernel>=6.28.0
gradio==4.44.0
pydantic==1.10.9
```

### setup.py
```python
from setuptools import setup, find_packages

setup(
    name="ursa-minor",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.5.1",
        "transformers>=4.46.1",
        "trl>=0.12.0",
        "datasets>=3.1.0",
        "accelerate>=1.0.1",
        "vllm>=0.6.4.post1",
        "wandb>=0.17.3",
    ],
)
```

### Environment Setup Script
```bash
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
```

## 3. Data Pipeline

### data/utils/io_utils.py
```python
import json
import hashlib
from typing import Any, Dict, List

def jload(file_path: str) -> Any:
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def jdump(data: Any, file_path: str) -> None:
    """Save data as JSON"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def question_hash(question: str) -> str:
    """Generate hash for question deduplication"""
    return hashlib.md5(question.encode('utf-8')).hexdigest()
```

### data/tokenization.py
```python
import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import Dict, Any, Optional

def mathcot_sft(
    download_data_path: str,
    upload_data_path: str,
    num_proc: int = 8,
    time_limit: bool = False,
    model_type: str = "qwen",
    rollout_path: Optional[str] = None,
    step_format: str = "nostepsnoanswer"
) -> Dataset:
    """
    Tokenize dataset for SFT training
    """
    # Load dataset
    dataset = load_dataset(download_data_path)
    
    # Load tokenizer
    if model_type == "qwen":
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    def format_example(example):
        """Format example for chat template"""
        question = example['question']
        thinking_trajectory = example['thinking_trajectories'][0]
        
        # Create chat format
        messages = [
            {"role": "system", "content": "You are Qwen developed by Alibaba. You should think step-by-step."},
            {"role": "user", "content": question}
        ]
        
        # Apply chat template for input
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Add thinking trajectory as assistant response
        text += thinking_trajectory
        
        return {"text": text}
    
    # Process dataset
    tokenized_dataset = dataset.map(
        format_example,
        num_proc=num_proc,
        desc="Tokenizing SFT data"
    )
    
    # Push to hub if upload path provided
    if upload_data_path:
        tokenized_dataset.push_to_hub(upload_data_path)
    
    return tokenized_dataset
```

### data/prepare_dataset.py
```python
#!/usr/bin/env python3
"""
Script to prepare and tokenize the s1K dataset for 7B model training
"""

import argparse
from datasets import load_dataset
from tokenization import mathcot_sft

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Ursa Minor 7B training")
    parser.add_argument("--input_dataset", type=str, default="qfq/s1K", 
                       help="Input dataset path")
    parser.add_argument("--output_dataset", type=str, default="ursa-minor/s1K-7b-tokenized",
                       help="Output tokenized dataset path")
    parser.add_argument("--num_proc", type=int, default=8,
                       help="Number of processes for tokenization")
    
    args = parser.parse_args()
    
    print(f"Tokenizing dataset: {args.input_dataset}")
    
    # Tokenize dataset
    tokenized_dataset = mathcot_sft(
        download_data_path=args.input_dataset,
        upload_data_path=args.output_dataset,
        num_proc=args.num_proc,
        model_type="qwen",
        step_format="nostepsnoanswer"
    )
    
    print(f"Tokenized dataset saved to: {args.output_dataset}")
    print(f"Dataset size: {len(tokenized_dataset['train'])}")

if __name__ == "__main__":
    main()
```

## 4. Training Infrastructure

### configs/training/fsdp_config_qwen_7b.json
```json
{
    "fsdp_transformer_layer_cls_to_wrap": [
        "Qwen2DecoderLayer"
    ],
    "fsdp_backward_prefetch": "backward_pre",
    "fsdp_forward_prefetch": false,
    "fsdp_sync_module_states": true,
    "fsdp_use_orig_params": false,
    "fsdp_cpu_ram_efficient_loading": true,
    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
    "fsdp_sharding_strategy": "FULL_SHARD"
}
```

### train/sft_7b.py
```python
import os
import warnings
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

import transformers
import trl
from datasets import load_dataset

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-7B-Instruct")
    block_size: int = field(default=16384)  # Reduced for 7B model
    wandb_project: Optional[str] = field(default="ursa-minor-7b")
    wandb_entity: Optional[str] = field(default="your-wandb-entity")
    train_file_path: Optional[str] = field(default='ursa-minor/s1K-7b-tokenized')
    
    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        if self.wandb_entity:
            os.environ['WANDB_ENTITY'] = self.wandb_entity

def train():
    # Parse arguments
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")
    
    # Load model with appropriate settings for 7B
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        use_cache=False
    )
    
    # Load dataset
    dataset = load_dataset(config.train_file_path)
    
    # Setup tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name, 
        use_fast=True
    )
    
    # Setup chat templates for Qwen
    instruction_template = "<|im_start|>user"
    response_template = "<|im_start|>assistant\n"
    tokenizer.pad_token = "<|fim_pad|>"
    
    # Data collator for completion-only training
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Configure training arguments
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    
    # Create trainer
    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
        args=args,
        data_collator=collator
    )
    
    # Train model
    trainer.train()
    
    # Save model and tokenizer
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()

if __name__ == "__main__":
    train()
```

### train/sft_7b.sh
```bash
#!/bin/bash
# Training script for 7B model - optimized for single node

uid="$(date +%Y%m%d_%H%M%S)"
base_model="Qwen/Qwen2.5-7B-Instruct"
lr=2e-5  # Slightly higher LR for 7B
epochs=5
weight_decay=1e-4
micro_batch_size=2  # Larger batch size for 7B
gradient_accumulation_steps=2
gpu_count=$(nvidia-smi -L | wc -l)

echo "Training Ursa Minor 7B with ${gpu_count} GPUs"

torchrun --nproc-per-node ${gpu_count} --master_port 12345 \
    train/sft_7b.py \
    --block_size=16384 \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path="ursa-minor/s1K-7b-tokenized" \
    --model_name=${base_model} \
    --warmup_ratio=0.05 \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="configs/training/fsdp_config_qwen_7b.json" \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="epoch" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="checkpoints/ursa-minor-7b-${uid}" \
    --push_to_hub=true \
    --hub_model_id="ursa-minor/7b-${uid}" \
    --save_only_model=True \
    --gradient_checkpointing=True \
    --report_to="wandb" \
    --run_name="ursa-minor-7b-${uid}"
```

## 5. Evaluation Pipeline

### eval/setup_evaluation.sh
```bash
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
```

### eval/evaluate_7b.py
```python
#!/usr/bin/env python3
"""
Evaluation script for Ursa Minor 7B model
"""

import subprocess
import argparse
from pathlib import Path

def run_evaluation(model_path: str, output_dir: str, tasks: str = "aime24_nofigures,openai_math,gpqa_diamond_openai"):
    """Run evaluation on specified tasks"""
    
    cmd = [
        "lm_eval",
        "--model", "vllm",
        "--model_args", f"pretrained={model_path},dtype=float32,tensor_parallel_size=1",
        "--tasks", tasks,
        "--batch_size", "auto",
        "--apply_chat_template",
        "--output_path", output_dir,
        "--log_samples",
        "--gen_kwargs", "max_gen_toks=16384,max_tokens_thinking=auto"
    ]
    
    print(f"Running evaluation command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Evaluation failed with error: {result.stderr}")
        return False
    
    print(f"Evaluation completed successfully. Results saved to: {output_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Evaluate Ursa Minor 7B model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--output_dir", type=str, default="results/evaluation",
                       help="Output directory for results")
    parser.add_argument("--tasks", type=str, 
                       default="aime24_nofigures,openai_math,gpqa_diamond_openai",
                       help="Evaluation tasks")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    success = run_evaluation(args.model_path, args.output_dir, args.tasks)
    
    if success:
        print("✅ Evaluation completed successfully")
    else:
        print("❌ Evaluation failed")
        exit(1)

if __name__ == "__main__":
    main()
```

## 6. Inference Pipeline

### inference/inference_7b.py
```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse
from typing import List

class UrsaMinor7BInference:
    def __init__(self, model_path: str):
        self.model = LLM(
            model_path,
            tensor_parallel_size=1,  # Single GPU for 7B
            dtype="float16"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Setup sampling parameters for thinking
        self.sampling_params = SamplingParams(
            max_tokens=16384,
            temperature=0.1,
            top_p=0.9,
            stop_token_ids=self.tokenizer.encode("<|im_end|>")
        )
    
    def generate_response(self, prompt: str) -> str:
        """Generate response with thinking"""
        messages = [
            {"role": "system", "content": "You are Qwen developed by Alibaba. You should think step-by-step."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        outputs = self.model.generate([text], self.sampling_params)
        return outputs[0].outputs[0].text
    
    def batch_generate(self, prompts: List[str]) -> List[str]:
        """Generate responses for multiple prompts"""
        formatted_prompts = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": "You are Qwen developed by Alibaba. You should think step-by-step."},
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_prompts.append(text)
        
        outputs = self.model.generate(formatted_prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]

def main():
    parser = argparse.ArgumentParser(description="Ursa Minor 7B Inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Input prompt")
    
    args = parser.parse_args()
    
    # Initialize inference
    model = UrsaMinor7BInference(args.model_path)
    
    # Generate response
    response = model.generate_response(args.prompt)
    
    print("=== PROMPT ===")
    print(args.prompt)
    print("\n=== RESPONSE ===")
    print(response)

if __name__ == "__main__":
    main()
```

## 7. Monitoring and Logging

### scripts/monitor_training.py
```python
#!/usr/bin/env python3
"""
Monitor training progress and log metrics
"""

import wandb
import argparse
import time
from pathlib import Path

def monitor_training(run_name: str, project: str = "ursa-minor-7b"):
    """Monitor training run via wandb"""
    
    api = wandb.Api()
    
    try:
        run = api.run(f"{project}/{run_name}")
        
        print(f"Monitoring run: {run_name}")
        print(f"Status: {run.state}")
        print(f"URL: {run.url}")
        
        # Get latest metrics
        history = run.history(keys=["train/loss", "train/learning_rate", "train/epoch"])
        
        if not history.empty:
            latest = history.iloc[-1]
            print(f"Latest Loss: {latest.get('train/loss', 'N/A')}")
            print(f"Learning Rate: {latest.get('train/learning_rate', 'N/A')}")
            print(f"Epoch: {latest.get('train/epoch', 'N/A')}")
        
    except Exception as e:
        print(f"Error monitoring run: {e}")

def main():
    parser = argparse.ArgumentParser(description="Monitor training run")
    parser.add_argument("--run_name", type=str, required=True,
                       help="Wandb run name")
    parser.add_argument("--project", type=str, default="ursa-minor-7b",
                       help="Wandb project name")
    
    args = parser.parse_args()
    
    monitor_training(args.run_name, args.project)

if __name__ == "__main__":
    main()
```

## 8. Complete Setup Script

### setup_complete_platform.sh
```bash
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
```

## 9. Training Execution

### Quick Start Commands
```bash
# 1. Setup everything
bash setup_complete_platform.sh

# 2. Start training
bash train/sft_7b.sh

# 3. Monitor training (in another terminal)
python scripts/monitor_training.py --run_name ursa-minor-7b-TIMESTAMP

# 4. Evaluate trained model
python eval/evaluate_7b.py --model_path checkpoints/ursa-minor-7b-TIMESTAMP

# 5. Run inference
python inference/inference_7b.py \
    --model_path checkpoints/ursa-minor-7b-TIMESTAMP \
    --prompt "How many r's are in the word 'raspberry'?"
```

## 10. Configuration Updates Needed

### Update These Files With Your Details:
1. **wandb entity**: Update `train/sft_7b.py` line with your wandb entity
2. **HuggingFace organization**: Update dataset paths in scripts to use your organization
3. **Model naming**: Update hub_model_id in training script to your desired naming convention

### Hardware Requirements:
- **Minimum**: 1x A100 (40GB) or equivalent
- **Recommended**: 2x A100 (40GB) for faster training
- **Memory**: 32GB+ system RAM
- **Storage**: 100GB+ for datasets, checkpoints, and results

### Training Time Estimates:
- **Single A100**: ~4-6 hours for 5 epochs
- **Dual A100**: ~2-3 hours for 5 epochs

This complete platform provides everything needed to train, evaluate, and deploy your Ursa Minor 7B distillation model using the s1K techniques and your existing dataset.