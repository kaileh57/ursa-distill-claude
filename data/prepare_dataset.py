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