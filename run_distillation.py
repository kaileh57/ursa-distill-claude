#!/usr/bin/env python3
"""
Main script to run the complete Ursa Minor 7B distillation pipeline
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

def run_command(cmd, description):
    """Run a shell command with error handling"""
    print(f"\n🔄 {description}")
    print(f"Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {description}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    
    print(f"✅ {description} completed successfully")
    if result.stdout:
        print(f"Output: {result.stdout}")
    
    return True

def setup_environment():
    """Setup the conda environment and dependencies"""
    print("🚀 Setting up environment...")
    
    # Check if conda is available
    result = subprocess.run("which conda", shell=True, capture_output=True)
    if result.returncode != 0:
        print("❌ Conda not found. Please install conda first.")
        return False
    
    # Setup environment
    commands = [
        "conda create -n ursa-minor python=3.10 -y",
        "conda run -n ursa-minor pip install -r requirements.txt",
        "conda run -n ursa-minor pip install -e ."
    ]
    
    for cmd in commands:
        if not run_command(cmd, f"Environment setup: {cmd}"):
            return False
    
    return True

def prepare_data():
    """Prepare and tokenize the dataset"""
    print("📊 Preparing dataset...")
    
    cmd = """conda run -n ursa-minor python data/prepare_dataset.py \
        --input_dataset "qfq/s1K" \
        --output_dataset "ursa-minor/s1K-7b-tokenized" \
        --num_proc 8"""
    
    return run_command(cmd, "Dataset preparation")

def setup_evaluation():
    """Setup evaluation environment"""
    print("🔍 Setting up evaluation...")
    
    return run_command("bash eval/setup_evaluation.sh", "Evaluation setup")

def run_training():
    """Start the training process"""
    print("🎯 Starting training...")
    
    # Check for GPU availability
    result = subprocess.run("nvidia-smi", shell=True, capture_output=True)
    if result.returncode != 0:
        print("⚠️  Warning: nvidia-smi not found. Training will fail without CUDA.")
        return False
    
    return run_command("conda run -n ursa-minor bash train/sft_7b.sh", "Model training")

def run_evaluation(model_path):
    """Run evaluation on the trained model"""
    print("📈 Running evaluation...")
    
    cmd = f"""conda run -n ursa-minor python eval/evaluate_7b.py \
        --model_path {model_path} \
        --output_dir results/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"""
    
    return run_command(cmd, "Model evaluation")

def test_inference(model_path):
    """Test inference on the trained model"""
    print("🧪 Testing inference...")
    
    test_prompt = "How many r's are in the word 'raspberry'?"
    cmd = f"""conda run -n ursa-minor python inference/inference_7b.py \
        --model_path {model_path} \
        --prompt "{test_prompt}" """
    
    return run_command(cmd, "Inference test")

def main():
    parser = argparse.ArgumentParser(description="Run Ursa Minor 7B distillation pipeline")
    parser.add_argument("--skip-setup", action="store_true", 
                       help="Skip environment setup")
    parser.add_argument("--skip-data", action="store_true",
                       help="Skip data preparation")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training (for testing)")
    parser.add_argument("--model-path", type=str,
                       help="Path to pre-trained model for evaluation/testing")
    parser.add_argument("--eval-only", action="store_true",
                       help="Only run evaluation on existing model")
    
    args = parser.parse_args()
    
    print("🌟 Starting Ursa Minor 7B Distillation Pipeline")
    print("=" * 50)
    
    # Create necessary directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results/training", exist_ok=True)
    os.makedirs("results/evaluation", exist_ok=True)
    os.makedirs("results/inference", exist_ok=True)
    
    success = True
    
    if args.eval_only:
        if not args.model_path:
            print("❌ Error: --model-path required for evaluation-only mode")
            sys.exit(1)
        
        success &= run_evaluation(args.model_path)
        success &= test_inference(args.model_path)
        
    else:
        # Full pipeline
        if not args.skip_setup:
            success &= setup_environment()
        
        if not args.skip_data and success:
            success &= prepare_data()
        
        if success:
            success &= setup_evaluation()
        
        if not args.skip_training and success:
            success &= run_training()
            
            # Find the latest checkpoint
            checkpoints_dir = Path("checkpoints")
            if checkpoints_dir.exists():
                latest_checkpoint = max(checkpoints_dir.glob("ursa-minor-7b-*"), 
                                      key=os.path.getctime, default=None)
                if latest_checkpoint:
                    model_path = str(latest_checkpoint)
                    success &= run_evaluation(model_path)
                    success &= test_inference(model_path)
                else:
                    print("❌ No checkpoints found after training")
                    success = False
        
        elif args.model_path:
            success &= run_evaluation(args.model_path)
            success &= test_inference(args.model_path)
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Pipeline completed successfully!")
        print("\nNext steps:")
        print("- Check wandb for training metrics")
        print("- Review evaluation results in results/evaluation/")
        print("- Test inference with your own prompts")
    else:
        print("❌ Pipeline failed. Check logs above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()