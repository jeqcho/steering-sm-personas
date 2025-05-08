# input: pii/single_cluster.jsonl and a list of folders e.g. [/processed_2_clusters,processed_25_clusters]
# output: cluster_X.jsonl where X is an integer from 0 upwards

# for each given folder, take the user_clusters.json file in them
# it is a Dict[str, int] which maps user ID to a cluster
# now go through each row of single_cluster.jsonl, where each row is a list of dicts.
# Look at the user_id of the last dict. Then write that row to the appropriate cluster
# do this for each folder

import hashlib
import json
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import os
from dotenv import load_dotenv

INPUT_FILE = Path(__file__).parent / "full_data" / "single_cluster.jsonl"
HOME_DIR = Path.home()

# Load environment variables from .env file
dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path)

# Get the secret from environment variables
SECRET = os.environ.get("HASH_SECRET")
if not SECRET:
    raise ValueError(
        "HASH_SECRET environment variable not found. Make sure to create .env file with HASH_SECRET."
    )


def load_user_clusters(folder_path: Path) -> Dict[str, int]:
    """Load the user_clusters.json file from the given folder."""
    user_clusters_file = folder_path / "user_clusters.json"
    if not user_clusters_file.exists():
        raise FileNotFoundError(
            f"User clusters file not found at: {user_clusters_file}"
        )

    with open(user_clusters_file, "r") as f:
        return json.load(f)


def process_folder(folder_path: str):
    """Process a folder, reading user clusters and writing chains to appropriate files."""
    path = Path(folder_path)
    folder_name = path.name
    # Output directly to the specified folder in home directory
    output_dir = Path(HOME_DIR / "cleaned" / folder_name)

    # Remove existing files if the directory exists
    if output_dir.exists():
        for file in output_dir.iterdir():
            file.unlink()
        print(f"Removed existing files from {output_dir}")
    else:
        output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Processing folder: {folder_path}")

    # First, copy over user_clusters.json to the output directory
    user_clusters_file = path / "user_clusters.json"

    assert user_clusters_file.exists()
    # Load the user clusters mapping
    user_clusters = load_user_clusters(path)

    # Create a mapping from cluster ID to output file
    cluster_files = {}

    # Count chains for progress bar
    with open(INPUT_FILE, "r") as f:
        total_chains = sum(1 for _ in f)

    # Process each chain in the single_cluster.jsonl
    with open(INPUT_FILE, "r") as f:
        for line in tqdm(f, total=total_chains, desc=f"Processing {folder_name}"):
            chain = json.loads(line)
            assert chain
            # Get the user_id from the last message in the chain
            last_message = chain[-1]
            user_id = last_message["user_id"]

            # Determine which cluster this user belongs to
            cluster_id = user_clusters.get(user_id)
            assert cluster_id is not None, (
                f"User ID {user_id} not found in user_clusters mapping. Cannot determine cluster assignment."
            )

            # drop the user_id
            dids = []
            for message in chain:
                dids.append(message.pop("user_id"))
            dump = json.dumps(chain, sort_keys=True)
            for i, message in enumerate(chain):
                hashed_did = hashlib.sha256(
                    f"{dids[i]}{dump}{SECRET}".encode()
                ).hexdigest()
                message["user_id"] = hashed_did

            line = json.dumps(chain) + "\n"
            # erase user_id and use anonymous_user_id
            # for message in chain:
            #     message["user_id"] = message["anonymized_user_id"]
            #     del message["anonymized_user_id"]

            # line = json.dumps(chain) + '\n'

            # Write the chain to the appropriate cluster file
            if cluster_id not in cluster_files:
                cluster_file = output_dir / f"cluster_{cluster_id}.jsonl"
                cluster_files[cluster_id] = open(cluster_file, "a")

            cluster_files[cluster_id].write(line)

    # Close all file handles
    for file_handle in cluster_files.values():
        file_handle.close()

    print(
        f"Completed processing {folder_name}. Created {len(cluster_files)} cluster files."
    )


def main():
    # Hardcoded list of folders to process (absolute paths from home directory)
    folders = [
        str(HOME_DIR / "processed_2_clusters"),
        str(HOME_DIR / "processed_25_clusters"),
        str(HOME_DIR / "processed_100_clusters"),
        str(HOME_DIR / "processed_1000_clusters"),
    ]

    for folder in folders:
        process_folder(folder)


if __name__ == "__main__":
    main()
