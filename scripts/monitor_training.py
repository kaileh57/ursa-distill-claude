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