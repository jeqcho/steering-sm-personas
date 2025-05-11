# Training Open-Source Models on Bluesky Data via Prefix Tuning

## Installation

```bash
cd ~/steering-sm-personas
python -m venv .venv
pip install -r requirements.txt
```

Get the user embeddings data from Compute Canada. We expect `~/caches/data_processing_user_data_cache.pkl`.
TODO: change this to PII removed.

## Run

Compute the embeddings for each cluster

```bash
cd ~/steering-sm-personas
nohup python -m src.compute_embeddings > logs/compute_embeddings.log 2>&1 &
```