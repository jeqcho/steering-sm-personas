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
import wandb

wandb.require("legacy-service")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

USERNAME = "jchooi"
USER_EMBEDDING_CACHE_FILE = f"/scratch/{USERNAME}/caches/data_processing_user_embedding_cache.pkl"

@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 2
    num_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    num_workers: int = 4
    checkpoint_dir: str = "/home/jchooi/scratch/checkpoints"
    save_every_n_epochs: int = 2
    gradient_accumulation_steps: int = 4

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize wandb
        wandb.init(
            entity="sandbox-persona",
            project="prefix-tune",
            config={
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "model_name": "meta-llama/Llama-3.2-3B-Instruct",
                "num_virtual_tokens": 20,
                "encoder_hidden_size": 1024
            }
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
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
        self.optimizer.zero_grad()
        
        # Initialize loss tracking
        recent_losses = []
        max_recent_losses = 4
        
        progress_bar = tqdm(self.train_dataloader, desc="Training", postfix={"loss": 0.0})
        for i, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            if (i + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Update recent losses
            batch_loss = loss.item() * self.config.gradient_accumulation_steps
            recent_losses.append(batch_loss)
            if len(recent_losses) > max_recent_losses:
                recent_losses.pop(0)
            
            # Calculate moving average
            moving_avg = sum(recent_losses) / len(recent_losses)
            
            # Update tqdm postfix with current moving average
            progress_bar.set_postfix({"loss": f"{moving_avg:.4f}"})
            progress_bar.write(f"Loss (last {len(recent_losses)} batches): {moving_avg:.4f}")
            
            # Log to wandb
            wandb.log({
                "batch_loss": batch_loss,
                "moving_avg_loss": moving_avg,
                "step": i
            })
            
            total_loss += batch_loss
        
        return total_loss / len(self.train_dataloader)
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.checkpoint_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Log checkpoint to wandb
        wandb.save(os.path.join(checkpoint_dir, "*"))
    
    def train(self):
        """Run the training loop."""
        logger.info("Starting training...")
        for epoch in range(self.config.num_epochs):
            avg_loss = self.train_epoch()
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}, Loss: {avg_loss:.4f}")
            
            # Log epoch metrics to wandb
            wandb.log({
                "epoch_loss": avg_loss,
                "epoch": epoch + 1
            })
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch + 1)
        
        logger.info("Training completed!")
        wandb.finish()

def main():
    # Initialize model and tokenizer
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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