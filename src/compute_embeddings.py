# this script generates the embeddings for the posts of a given cluster file
# clusters are in ~/cleaned/
from pathlib import Path
import json
import numpy as np
import pickle
import re
from tqdm import tqdm

import os
import torch
from transformers import AutoTokenizer, AutoModel
from SM_based_personas.data_processing.src.utils import logger, MODEL_NAME, EMBEDDING_CACHE_FILE

class EmbeddingManager:
    def __init__(self, batch_size=32):
        self.tokenizer = None
        self.model = None
        self.device = None
        self.batch_size = batch_size

        logger.info(f"Process {torch.multiprocessing.current_process().pid} loading embedding model")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Process {torch.multiprocessing.current_process().pid} using device: {self.device}")
    
    def get_embeddings(self, text: list[str]):
        """Get embedding for a list of texts using batching"""
        # Check if tokenizer and model are initialized
        if self.tokenizer is None or self.model is None:
            raise ValueError("Tokenizer or model not initialized. Make sure EmbeddingManager is properly initialized.")
        
        # Process texts in batches to avoid memory issues
        all_embeddings = []
        
        # Create batches with tqdm progress bar
        for i in tqdm(range(0, len(text), self.batch_size), desc="Computing embeddings", leave=False):
            batch = text[i:i + self.batch_size]
            
            # Process batch
            inputs = self.tokenizer(
                batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            ).to(self.device)
            
            # Get embedding from model
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use CLS token embedding (first token of last hidden state)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batch results
        if all_embeddings:
            return np.vstack(all_embeddings)
        else:
            return np.array([])

def extract_cluster_id(filename: str) -> int:
    """Extract numeric cluster ID from a filename like 'cluster_12.jsonl'."""
    match = re.search(r"cluster_(\d+)", filename)
    assert match
    return int(match.group(1))  # Get just the numeric part


def compute_cluster_embedding(cluster_path: Path, manager: EmbeddingManager):
    """Compute embedding for a single cluster file.

    For each line in the file (which is a JSON list of dictionaries),
    get the last dictionary, extract its text, compute an embedding,
    and return the average of all embeddings.
    """
    with open(cluster_path, "r") as f:
        lines = f.readlines()
        all_text:list[str] = []

        # Process each line in the file
        for line in tqdm(lines, desc=f"Loading {cluster_path.name}", leave=False):
            # Each line is a JSON string representing a list of dictionaries
            data = json.loads(line)

            assert isinstance(data, list)

            # Get the last dictionary in the list
            last_item = data[-1]
            assert isinstance(last_item, dict)

            # Check if it has no "actions" key and compute embedding for "text"
            if "actions" not in last_item:
                assert "text" in last_item
                text = last_item["text"]
                all_text.append(text)
                
        embeddings = manager.get_embeddings(all_text)
        avg_embedding = np.mean(embeddings, axis=0)
        return avg_embedding


def compute_embeddings_in_folder(folder_path: Path, manager: EmbeddingManager):
    """Process all cluster files in a folder and save their embeddings."""
    # Get all jsonl files in the folder
    jsonl_files = list(folder_path.glob("*.jsonl"))

    # Map of cluster_id to embedding
    cluster_embeddings = [0] * len(jsonl_files)

    for jsonl_file in tqdm(jsonl_files, desc=f"Processing {folder_path.name}"):
        embedding = compute_cluster_embedding(jsonl_file, manager)

        # Extract just the numeric part from the filename
        filename = jsonl_file.stem  # Get filename without extension
        cluster_id = extract_cluster_id(filename)

        cluster_embeddings[cluster_id] = embedding

    # Save embeddings to pickle file
    save_embeddings(folder_path, cluster_embeddings)


def save_embeddings(folder_path: Path, cluster_embeddings: list):
    """Save cluster embeddings to a pickle file."""
    output_path = folder_path / "embeddings.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(cluster_embeddings, f)

    print(f"Saved embeddings for {len(cluster_embeddings)} clusters to {output_path}")


if __name__ == "__main__":
    base_path = "~/cleaned"
    folder_paths = [
        # f"{base_path}/processed_2_clusters",
        # f"{base_path}/processed_25_clusters",
        f"{base_path}/processed_100_clusters",
        f"{base_path}/processed_1000_clusters",
    ]
    folder_paths = [Path(path).expanduser() for path in folder_paths]

    manager = EmbeddingManager(batch_size=512)

    for folder_path in tqdm(folder_paths, desc="Processing folders"):
        compute_embeddings_in_folder(folder_path, manager)
