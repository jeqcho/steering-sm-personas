# read ../processed_25_clusters/* and load all the jsonl
# each jsonl has the following form: [message,message,message,...]
# where message is a a dict
# each message has the following keys
# user_id: str, text: str, actions: Dict, unix_epoch:int
# the message will either have the text key or the actions key, never both, but always exactly one.
# call each row of the jsonl has a chain, and give it a chain_id: int.
# now make a dataframe
# the dataframe should have the cols user_id: str, text: str, actions: str, unix_epoch:int.
# write the dataframe to processed_25_clusters as a easy to load format

from typing import List, Dict, Any
import json
import glob
import pandas as pd
from pathlib import Path

def load_and_process_jsonl(file_path: Path, chain_id_start: int) -> tuple[int, List[Dict[str, Any]]]:
    messages: List[Dict[str, Any]] = []
    with open(file_path, 'r') as f:
        data: List[List[Dict[str, Any]]] = []
        for line in f:
            data.append(json.loads(line))
        for chain_idx, chain in enumerate(data, start=chain_id_start):
            for message in chain:
                # Create a row with either text or actions
                row: Dict[str, Any] = {
                    'user_id': message['user_id'],
                    'unix_epoch': message['unix_epoch'],
                    'chain_id': chain_idx,
                    'text': message.get('text', ''),
                    'actions': json.dumps(message.get('actions', {})) if 'actions' in message else ''
                }
                messages.append(row)
        num_chains: int = len(data)
    return num_chains, messages

def main() -> None:
    # Get all jsonl files in the processed_25_clusters directory
    input_dir: Path = Path.home() / 'processed_25_clusters'
    output_dir: Path = Path.home() / 'all_messages'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_messages: List[Dict[str, Any]] = []
    chain_id_counter: int = 0
    
    # Process each jsonl file
    for file_path in input_dir.glob('*.jsonl'):
        num_chains, messages = load_and_process_jsonl(file_path, chain_id_counter)
        all_messages.extend(messages)
        chain_id_counter += num_chains
    
        # Log the number of messages loaded from each file
        print(f"Loaded {len(messages)} messages from {file_path.name}")
    
        # Log running total of messages
        print(f"Running total: {len(all_messages)} messages across {chain_id_counter} chains")
    
    print("Loading json done!")
    # Create DataFrame
    df: pd.DataFrame = pd.DataFrame(all_messages)
    
    # Ensure correct column types
    df['user_id'] = df['user_id'].astype(str)
    df['unix_epoch'] = df['unix_epoch'].astype(int)
    df['chain_id'] = df['chain_id'].astype(int)
    
    # Save DataFrame in parquet format (efficient and easily loadable)
    output_path: Path = output_dir / 'merged_messages.parquet'
    df.to_parquet(output_path, index=False)
    
    # Also save a CSV version for easy viewing if needed
    # csv_output_path: Path = output_dir / 'merged_messages.csv'
    # df.to_csv(csv_output_path, index=False)
    
    print(f"Processed {len(df)} messages across {df['chain_id'].nunique()} chains")
    # print(f"Data saved to {output_path} and {csv_output_path}")

if __name__ == "__main__":
    main()