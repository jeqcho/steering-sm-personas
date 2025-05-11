from pathlib import Path
from typing import Any
from peft.mixed_model import PeftMixedModel
from peft.peft_model import PeftModel
import torch
from torch.utils.data import DataLoader
from peft import get_peft_model, PrefixTuningConfig, TaskType
from tqdm import tqdm
import os
from dataclasses import dataclass
import wandb
from shared import (
    ConversationDataset,
    load_model_and_tokenizer,
    collate_fn,
    logger
)
from huggingface_hub import login
from dotenv import load_dotenv

# wandb.require("legacy-service")

@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 2
    num_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    num_workers: int = 4
    checkpoint_dir: str = "/home/ubuntu/sandbox-persona/checkpoints"
    save_every_n_steps: int = 100  # Save checkpoint every N steps
    gradient_accumulation_steps: int = 4

class PrefixTrainer:
    """Trainer for the PEFT model."""
    
    def __init__(
        self,
        model: PeftModel | PeftMixedModel,
        train_dataset: ConversationDataset,
        test_dataset: ConversationDataset,  # Add test dataset
        config: TrainingConfig,
        tokenizer: Any,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset  # Store test dataset
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
                "num_virtual_tokens": 10,
                "encoder_hidden_size": 1024
            }
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup dataloaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.num_workers
        )
        
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
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
    
    def evaluate(self) -> float:
        """Evaluate the model on the test set."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.test_dataloader)
        logger.info(f"Evaluation Loss: {avg_loss:.4f}")
        
        # Log to wandb
        wandb.log({
            "eval_loss": avg_loss
        })
        
        return avg_loss

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()
        
        # Initialize loss tracking
        recent_losses = []
        max_recent_losses = 4
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch + 1}", postfix={"loss": 0.0})
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
                "step": i + epoch * len(self.train_dataloader)
            })
            
            total_loss += batch_loss
            
            # Save checkpoint every N steps
            if (i + 1) % self.config.save_every_n_steps == 0:
                self.save_checkpoint(f"step_{i+1}")
            
            # Evaluate every 250 steps
            if (i + 1) % 250 == 0:
                eval_loss = self.evaluate()
                logger.info(f"Step {i + 1}: Evaluation Loss = {eval_loss:.4f}")
                self.model.train()  # Switch back to training mode
        
        return total_loss / len(self.train_dataloader)
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.checkpoint_dir, f"checkpoint-{checkpoint_name}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Log checkpoint to wandb
        wandb.save(os.path.join(checkpoint_dir, "*"))
    
    def train(self):
        """Run the training loop."""
        logger.info("Starting training...")
        for epoch in range(self.config.num_epochs):
            avg_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}, Loss: {avg_loss:.4f}")
            
            # Log epoch metrics to wandb
            wandb.log({
                "epoch_loss": avg_loss,
                "epoch": epoch + 1
            })
        
        logger.info("Training completed!")
        wandb.finish()

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Login to Hugging Face
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env file. Please add it to your .env file.")
    login(token=hf_token)
    
    # Login to WandB
    wandb_token = os.getenv("WANDB_API_KEY")
    if not wandb_token:
        raise ValueError("WANDB_API_KEY not found in .env file. Please add it to your .env file.")
    wandb.login(key=wandb_token)
    
    # Load base model and tokenizer
    base_model, tokenizer = load_model_and_tokenizer()
    
    # Setup PEFT config
    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=10,
        encoder_hidden_size=1024  # Match user embedding dimension
    )
    
    # Get PEFT model
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    
    # Create datasets
    cluster_folder_path = Path("/home/ubuntu/cleaned/processed_100_clusters/")
    train_dataset = ConversationDataset(cluster_folder_path, tokenizer, split='train', test_size=0.001)
    test_dataset = ConversationDataset(cluster_folder_path, tokenizer, split='test', test_size=0.001)
    
    # Training configuration
    config = TrainingConfig(batch_size=5)
    
    # Initialize trainer
    trainer = PrefixTrainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=config,
        tokenizer=tokenizer
    )
    
    # Train
    trainer.train()

if __name__ == "__main__":
    main()