# this file loads ~/caches/embedding_cache.pkl and outputs its structure
# be careful since it is 5GB

import pickle
import os

from SM_based_personas.data_processing.src.embedding import EmbeddingManager

CACHE_PATH = os.path.expanduser('~/caches/embedding_cache.pkl')

def main():
    print(f"Loading: {CACHE_PATH}")
    with open(CACHE_PATH, 'rb') as f:
        obj = pickle.load(f)
    print(f"Top-level type: {type(obj)}")
    if isinstance(obj, dict):
        print(f"Top-level dict keys (up to 20): {list(obj.keys())[:20]}")
        print(f"Number of keys: {len(obj)}")
        # Print info about the first item
        first_key = next(iter(obj))
        first_val = obj[first_key]
        print(f"First key: {first_key}")
        print(f"Type of first value: {type(first_val)}")
        if hasattr(first_val, 'shape'):
            print(f"Shape of first value: {getattr(first_val, 'shape', None)}")
        if hasattr(first_val, 'dtype'):
            print(f"Dtype of first value: {getattr(first_val, 'dtype', None)}")
    elif isinstance(obj, list):
        print(f"Top-level list length: {len(obj)}")
        print(f"Type of first element: {type(obj[0])}")
    else:
        print(f"Object summary: {str(obj)[:500]}")

def embed_example_text():
    manager = EmbeddingManager()
    text = "This is an example sentence to embed."
    embedding = manager.get_embedding(text)
    print("Embedding shape:", embedding.shape)
    print("First 10 values:", embedding[:10])

if __name__ == "__main__":
    main()
    embed_example_text()