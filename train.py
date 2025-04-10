from typing import List, Dict, Any
import json
import torch
from torch.utils.data import Dataset
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from spm import SPM
 
from utils import convert_chain_to_text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: Any):
        self.tokenizer = tokenizer
        self.examples: List[str] = []
        
        with open(file_path, 'r') as f:
            for line in f:
                chain = json.loads(line)
                text = convert_chain_to_text(chain)
                self.examples.append(text)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.examples[idx]
        
        # Tokenize the full text
        tokenized = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0)
        }

class SPMPrefixTrainer(Trainer):
    def __init__(self, spm: SPM, base_model: Any, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spm = spm
        self.base_model = base_model
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # Get the SPM's output for the current batch
        batch_size = inputs["input_ids"].shape[0]
        spm_input = torch.ones(batch_size, 1, device=inputs["input_ids"].device)  # Input is just ones
        prefix = self.spm(spm_input)  # Shape: [batch_size, T, L×2×D/H]
        
        # Reshape prefix for each layer
        prefix = prefix.view(batch_size, self.spm.num_virtual_tokens, self.spm.num_layers, 2, -1)
        
        # Forward pass through base model with prefix
        outputs = self.base_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            prefix=prefix
        )
        
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

def main():
    # Initialize model and tokenizer
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        device_map="auto"
    )
    
    # Freeze Qwen's weights
    for param in model.parameters():
        param.requires_grad = False
    
    # Initialize SPM
    num_layers = len(model.transformer.h)  # Number of transformer layers
    token_dim = model.config.hidden_size    # Token dimension
    num_heads = model.config.num_attention_heads  # Number of attention heads
    num_virtual_tokens = 1  # Number of virtual tokens
    
    spm = SPM(
        num_layers=num_layers,
        token_dim=token_dim,
        num_heads=num_heads,
        num_virtual_tokens=num_virtual_tokens
    ).to(model.device)
    
    # Create dataset
    dataset = ConversationDataset("downloaded_clusters/cluster_0.jsonl", tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,  # Train for 10 epochs
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=0.001,  # SPM learning rate
        weight_decay=0.001,   # SPM weight decay
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        fp16=True,
        gradient_checkpointing=True,
    )
    
    # Initialize trainer with SPM
    trainer = SPMPrefixTrainer(
        spm=spm,
        base_model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed!")

if __name__ == "__main__":
    main()