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