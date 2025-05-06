import json
import os
import hashlib
import argparse

parser = argparse.ArgumentParser(description='Compute the statistics of clusters')
parser.add_argument("--folder_name", type=str, help="Name of the folder that contains the clusters", default="/home/s4yor1/scratch/pii_removed/processed_25_clusters")

args = parser.parse_args()

DATASET_PATH = args.folder_name

with open("secret_hash_did.tok", 'r') as f:
    SECRET = f.read().strip()

# Load the cluster map
with open(os.path.join(DATASET_PATH, "user_clusters.json"), 'r') as f:
    unhashed_cluster_map = json.load(f)

hashed_cluster_map = {}

for file in os.listdir(DATASET_PATH):
    if file.endswith(".jsonl"):
        with open(os.path.join(DATASET_PATH, file), 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f.readlines()]
    
        cluster_id = file.split(".")[0].split("_")[-1]
        
        # Safely hash the DID by taking
        # contatenate did + secret + json dump of chain without did
        
        for chain in data:
            dids = []
            for message in chain:
                dids.append(message.pop("user_id"))
            dump = json.dumps(chain, sort_keys=True)
            for i, message in enumerate(chain):
                hashed_did = hashlib.sha256(f"{dids[i]}{dump}{SECRET}".encode()).hexdigest()
                message["user_id"] = hashed_did
                chain[i] = message
                hashed_cluster_map[hashed_did] = unhashed_cluster_map[dids[i]]
        
        # Create the output directory if it doesn't exist
        output_dir = DATASET_PATH + "_hashed"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, file), 'w', encoding='utf-8') as f:
            for chain in data:
                f.write(json.dumps(chain) + "\n")
        print(f"Processed {file}")
        
# Save the hashed cluster map
with open(os.path.join(DATASET_PATH + "_hashed", "user_clusters.json"), 'w') as f:
    json.dump(hashed_cluster_map, f, indent=4)