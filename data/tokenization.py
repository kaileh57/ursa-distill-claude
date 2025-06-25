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