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