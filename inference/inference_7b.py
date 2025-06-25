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