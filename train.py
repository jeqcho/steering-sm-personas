from typing import List, Dict, Any, Tuple, Optional
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_config, get_peft_model, PrefixTuningConfig, TaskType
from numpy.typing import NDArray
import numpy as np
import pickle
from utils import convert_chain_to_text, get_subject_user_id
import logging
from tqdm import tqdm
import os
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

USERNAME = "jchooi"
USER_EMBEDDING_CACHE_FILE = f"/scratch/{USERNAME}/caches/data_processing_user_embedding_cache.pkl"

@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 4
    num_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    num_workers: int = 4
    checkpoint_dir: str = "./results"
    save_every_n_epochs: int = 2
    num_virtual_tokens: int = 20  # Number of virtual tokens for prefix tuning

class ConversationDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: Any):
        self.tokenizer = tokenizer
        self.examples: List[Tuple[str, NDArray[np.float32]]] = []
        
        # read user embeddings
        user_embeddings = {}
        with open(USER_EMBEDDING_CACHE_FILE, 'rb') as f:
            user_embeddings = pickle.load(f)
        
        # read actual test
        with open(file_path, 'r') as f:
            for line in f:
                chain = json.loads(line)
                user_id = get_subject_user_id(chain)
                embeddings = user_embeddings[user_id]
                text = convert_chain_to_text(chain)
                payload = (text, embeddings)
                self.examples.append(payload)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text, embeddings = self.examples[idx]
        embeddings = torch.tensor(embeddings)
        
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
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": tokenized["input_ids"].squeeze(0)  # For language modeling
        }

def collate_fn(batch):
    """Custom collate function to handle our dataset format."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

class PrefixTrainer:
    """Trainer for the PEFT model."""
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        train_dataset: Dataset,
        config: TrainingConfig,
        tokenizer: Any,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.config = config
        self.tokenizer = tokenizer
        self.device = model.device
        
        # Setup dataloader
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.num_workers
        )
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_dataloader, desc="Training"):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_dataloader)
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.checkpoint_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def train(self):
        """Run the training loop."""
        logger.info("Starting training...")
        for epoch in range(self.config.num_epochs):
            avg_loss = self.train_epoch()
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}, Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch + 1)
        
        logger.info("Training completed!")

def main():
    # Initialize model and tokenizer
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Setup PEFT config
    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,
        encoder_hidden_size=1024  # Match user embedding dimension
    )
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Create dataset
    dataset = ConversationDataset("downloaded_clusters/cluster_0.jsonl", tokenizer)
    
    # Training configuration
    config = TrainingConfig()
    
    # Initialize trainer
    trainer = PrefixTrainer(
        model=model,
        train_dataset=dataset,
        config=config,
        tokenizer=tokenizer
    )
    
    # Train
    trainer.train()

if __name__ == "__main__":
    main()