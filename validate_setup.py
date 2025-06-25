#!/usr/bin/env python3
"""
Validation script to check if the distillation system is properly set up
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} NOT FOUND")
        return False

def check_executable(filepath, description):
    """Check if a file is executable"""
    if Path(filepath).exists() and os.access(filepath, os.X_OK):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} NOT EXECUTABLE")
        return False

def check_python_import(module_name, description):
    """Check if a Python module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError:
        print(f"❌ {description}: {module_name} NOT AVAILABLE")
        return False

def check_command(command, description):
    """Check if a command is available"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description}: {command}")
            return True
        else:
            print(f"❌ {description}: {command} FAILED")
            return False
    except Exception as e:
        print(f"❌ {description}: {command} ERROR - {e}")
        return False

def main():
    print("🔍 Validating Ursa Minor 7B Distillation Setup")
    print("=" * 50)
    
    all_good = True
    
    # Check directory structure
    print("\n📁 Checking directory structure...")
    directories = [
        "data", "train", "eval", "inference", "scripts", "configs", 
        "checkpoints", "results", "data/utils", "configs/training"
    ]
    
    for directory in directories:
        if Path(directory).exists():
            print(f"✅ Directory: {directory}")
        else:
            print(f"❌ Directory: {directory} NOT FOUND")
            all_good = False
    
    # Check core files
    print("\n📄 Checking core files...")
    core_files = [
        ("requirements.txt", "Requirements file"),
        ("setup.py", "Setup file"),
        ("run_distillation.py", "Main execution script"),
        ("setup_complete_platform.sh", "Platform setup script"),
        ("data/prepare_dataset.py", "Data preparation script"),
        ("train/sft_7b.py", "Training script"),
        ("train/sft_7b.sh", "Training shell script"),
        ("eval/evaluate_7b.py", "Evaluation script"),
        ("inference/inference_7b.py", "Inference script"),
        ("configs/training/fsdp_config_qwen_7b.json", "FSDP config"),
    ]
    
    for filepath, description in core_files:
        if not check_file_exists(filepath, description):
            all_good = False
    
    # Check executable permissions
    print("\n🔧 Checking executable permissions...")
    executables = [
        ("run_distillation.py", "Main script"),
        ("setup_complete_platform.sh", "Setup script"),
        ("data/prepare_dataset.py", "Data prep script"),
        ("train/sft_7b.sh", "Training script"),
        ("eval/evaluate_7b.py", "Evaluation script"),
        ("inference/inference_7b.py", "Inference script"),
    ]
    
    for filepath, description in executables:
        if not check_executable(filepath, description):
            all_good = False
    
    # Check Python dependencies (basic ones)
    print("\n🐍 Checking Python dependencies...")
    dependencies = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("numpy", "NumPy"),
        ("json", "JSON (built-in)"),
        ("subprocess", "Subprocess (built-in)"),
        ("argparse", "Argparse (built-in)"),
    ]
    
    for module, description in dependencies:
        if not check_python_import(module, description):
            if module not in ["torch", "transformers", "datasets"]:  # Built-ins should always work
                all_good = False
    
    # Check system commands
    print("\n💻 Checking system commands...")
    commands = [
        ("python --version", "Python"),
        ("git --version", "Git"),
        ("which conda || which mamba", "Conda/Mamba"),
    ]
    
    for command, description in commands:
        check_command(command, description)  # Don't fail on system commands
    
    # Check GPU availability (optional)
    print("\n🖥️  Checking GPU availability...")
    gpu_available = check_command("nvidia-smi", "NVIDIA GPU")
    if not gpu_available:
        print("⚠️  Warning: No NVIDIA GPU detected. Training will not work without CUDA.")
    
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 All core components are properly set up!")
        print("\n✅ Ready to run distillation pipeline")
        print("Next steps:")
        print("1. python run_distillation.py")
        print("2. Or follow README.md for manual execution")
    else:
        print("❌ Some components are missing or not properly configured")
        print("Please check the errors above and run setup again.")
        sys.exit(1)

if __name__ == "__main__":
    main()