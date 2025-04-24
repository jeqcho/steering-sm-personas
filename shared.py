from typing import List, Dict, Any, Tuple, Optional
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from numpy.typing import NDArray
import numpy as np
import pickle
from utils import convert_chain_to_text, get_subject_user_id
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shared constants
USERNAME = "jchooi"
USER_EMBEDDING_CACHE_FILE = f"/home/ubuntu/sandbox-persona/caches/data_processing_user_embedding_cache.pkl"
CLUSTER_POST_FILES = [
    "downloaded_clusters/cluster_0.jsonl",
    "downloaded_clusters/cluster_1.jsonl",
    "downloaded_clusters/cluster_2.jsonl"
]

class ConversationDataset(Dataset):
    def __init__(self, file_paths: List[str], tokenizer: Any, split: Optional[str] = None, test_size: float = 0.2, random_state: int = 42):
        self.tokenizer = tokenizer
        self.examples: List[Tuple[str, str, NDArray[np.float32]]] = []  # (full_text, assistant_text, embeddings)
        
        # read user embeddings
        user_embeddings = {}
        with open(USER_EMBEDDING_CACHE_FILE, 'rb') as f:
            user_embeddings = pickle.load(f)
        
        # read actual test
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                for line in f:
                    chain = json.loads(line)
                    user_id = get_subject_user_id(chain)
                    embeddings = user_embeddings[user_id]
                    full_text = convert_chain_to_text(chain)
                    
                    # Split text to get assistant's response
                    parts = full_text.split("<|im_start|>assistant")
                    if len(parts) != 2:
                        continue  # Skip malformed examples
                    assistant_text = "<|im_start|>assistant" + parts[1]  # Include separator in assistant part
                    
                    payload = (full_text, assistant_text, embeddings)
                    self.examples.append(payload)
        
        # Perform train/test split if requested
        if split is not None:
            if split not in ['train', 'test']:
                raise ValueError("split must be either 'train' or 'test'")
            
            train_examples, test_examples = train_test_split(
                self.examples,
                test_size=test_size,
                random_state=random_state
            )
            self.examples = train_examples if split == 'train' else test_examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        full_text, assistant_text, embeddings = self.examples[idx]
        embeddings = torch.tensor(embeddings)
        
        # Tokenize the full text for input
        tokenized = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )
        
        # Create labels where we only compute loss on assistant's response
        # First, find where the assistant's text starts in the full text
        full_tokens = self.tokenizer.encode(full_text)
        assistant_tokens = self.tokenizer.encode(assistant_text)
        
        # Create labels with -100 for tokens before assistant's response
        labels = tokenized["input_ids"].clone()
        assistant_start_idx = len(full_tokens) - len(assistant_tokens)
        labels[:, :assistant_start_idx] = -100  # Set to -100 to ignore in loss
        
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),  # Only compute loss on assistant's response
            "text": full_text  # Store original text for reference
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

def load_model_and_tokenizer(model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
    """Load base model and tokenizer with common settings."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    return model, tokenizer 