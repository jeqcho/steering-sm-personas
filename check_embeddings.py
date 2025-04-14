import pickle
import numpy as np
import torch
from typing import Dict, Any

# Global constants
USERNAME = "jchooi"
USER_EMBEDDING_DIM = 1024  # Dimension of user embeddings
USER_EMBEDDING_CACHE_FILE = f"/scratch/{USERNAME}/caches/data_processing_user_embedding_cache.pkl"

def load_embeddings() -> Dict[str, np.ndarray]:
    """Load user embeddings from cache file."""
    with open(USER_EMBEDDING_CACHE_FILE, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

def main():
    # Load embeddings
    embeddings = load_embeddings()
    
    # Print basic statistics
    print(f"Number of users: {len(embeddings)}")
    
    # Check dimensions of first few embeddings
    for i, (user_id, emb) in enumerate(list(embeddings.items())[:5]):
        print(f"\nUser {user_id}:")
        print(f"Shape: {emb.shape}")
        print(f"Type: {emb.dtype}")
        print(f"Min value: {emb.min():.4f}")
        print(f"Max value: {emb.max():.4f}")
        print(f"Mean value: {emb.mean():.4f}")
        print(f"Std value: {emb.std():.4f}")
        
        # Verify embedding dimension
        if emb.shape[-1] != USER_EMBEDDING_DIM:
            print(f"WARNING: Expected embedding dimension {USER_EMBEDDING_DIM}, got {emb.shape[-1]}")
        
        if i >= 4:  # Only check first 5 users
            break
    
    # Check if all embeddings have same dimension
    dims = set(emb.shape for emb in embeddings.values())
    if len(dims) == 1:
        print(f"\nAll embeddings have the same dimension: {dims.pop()}")
    else:
        print("\nWARNING: Not all embeddings have the same dimension!")
        print("Dimensions found:", dims)

if __name__ == "__main__":
    main() 