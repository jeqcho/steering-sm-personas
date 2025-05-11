"""
Write results to outputs/
"""

from pathlib import Path
import torch
from peft import PeftModel
import random
import os
from datetime import datetime
from shared import (
    ConversationDataset,
    load_model_and_tokenizer,
    logger
)
from tqdm import tqdm

def generate_completions(
    checkpoint_path: str = "/home/ubuntu/sandbox-persona/checkpoints/checkpoint-step_500",
    examples_per_cluster: int = 5,
    output_dir: str = "outputs"
):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamped output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"output_{timestamp}.txt")
    
    # Load base model and tokenizer
    base_model, tokenizer = load_model_and_tokenizer()
    
    # Load PEFT model from checkpoint
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    
    # Generate completions
    with open(output_file, 'w') as f:
        cluster_folder_path = Path("/home/ubuntu/cleaned/processed_100_clusters/")
        file_paths = list(cluster_folder_path.glob("*.jsonl"))
        # Process each cluster file separately
        for cluster_file in tqdm(file_paths):
            # Create dataset for this cluster
            dataset = ConversationDataset([cluster_file], tokenizer)
            
            # Select random examples from this cluster
            if len(dataset) < examples_per_cluster:
                logger.warning(f"Cluster {cluster_file} has fewer than {examples_per_cluster} examples. Using all available examples.")
                indices = range(len(dataset))
            else:
                indices = random.sample(range(len(dataset)), examples_per_cluster)
            
            # Write cluster header
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Cluster: {cluster_file}\n")
            f.write(f"{'=' * 80}\n\n")
            
            # Generate completions for selected examples
            for idx in indices:
                example = dataset[idx]
                input_text: str = example["text"]
                
                # Split input into conditioning and target parts
                # Assuming format is <|im_start|>user_1 ... <|im_end|> <|im_start|>assistant
                parts = input_text.rsplit("<|start_header_id|>assistant<|end_header_id|>\n\n", 1)
                    
                conditioning_text = parts[0] + "<|start_header_id|>assistant<|end_header_id|>\n\n"
                
                # Tokenize only the conditioning part
                inputs = tokenizer(conditioning_text, return_tensors="pt", padding=True, truncation=True, max_length=2048, add_special_tokens=False)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Generate completion
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=600,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id
                    )
                
                # Decode completion
                completion = tokenizer.decode(outputs[0], skip_special_tokens=False)
                
                # Write to file with nice formatting
                f.write("Original Message:\n")
                f.write("-" * 40 + "\n")
                f.write(input_text + "\n\n")
                f.write("Generated Completion:\n")
                f.write("-" * 40 + "\n")
                f.write(completion + "\n\n")
                f.write("-" * 80 + "\n\n")
    
    logger.info(f"Generated completions written to {output_file}")

if __name__ == "__main__":
    generate_completions() 