# input full_data/pii_dataset_tags.csv
# output: full_data/single_cluster.jsonl

# loop through the rows of pii_dataset_tags (as a df)
# the columns are user_id,unix_epoch,chain_id,text,actions,full_text,lowest_score,presidio_batch_output,scrubbed_output,final_flag
# each row is a message. you should recrete it into a dict, which has user_id, unix_epoch, text, actions. The text value should be scrubbed_output.
# the message should have exactly one of text and actions.
# note that each row in the df can be clumped together into a chain. And these rows are consecutive rows. They have the same chain id

# assert that for each message there's only one of text or actions.

# write this to output

import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

INPUT_FILE = Path(__file__).parent / 'full_data' / 'pii_dataset_tags.csv'
OUTPUT_FILE = Path(__file__).parent / 'full_data' / 'single_cluster.jsonl'

# Columns: user_id,unix_epoch,chain_id,text,actions,full_text,lowest_score,presidio_batch_output,scrubbed_output,final_flag

def process_and_write():
    last_chain_id = None
    current_chain = []
    first_epoch = None
    with open(OUTPUT_FILE, 'w') as fout:
        chunk_iter = pd.read_csv(INPUT_FILE, chunksize=100_000)
        total_chunks =  12*1e6 / 100_000
        for chunk in tqdm(chunk_iter, desc='Chunks', total=total_chunks):
            for row in tqdm(chunk.itertuples(index=False), total=len(chunk), leave=False, desc='Rows in chunk'):
                row_dict = row._asdict() if hasattr(row, '_asdict') else dict(zip(chunk.columns, row))
                chain_id = row_dict['chain_id']
                
                # If chain_id changes, reset the first_epoch and write the previous chain
                if last_chain_id is not None and chain_id != last_chain_id:
                    # Before writing, adjust all unix_epoch values to be relative to the first message
                    fout.write(json.dumps(current_chain) + '\n')
                    current_chain = []
                    first_epoch = None
                
                # Record the first epoch time in this chain if this is the first message
                if first_epoch is None:
                    first_epoch = int(row_dict['unix_epoch'])
                
                # Calculate relative unix_epoch
                relative_epoch = int(row_dict['unix_epoch']) - first_epoch
                
                msg = {
                    'user_id': row_dict['user_id'],
                    'unix_epoch': relative_epoch,
                }
                
                has_actions = pd.notnull(row_dict['actions']) and str(row_dict['actions']).strip() != ''
                has_text = pd.notnull(row_dict['scrubbed_output']) and str(row_dict['scrubbed_output']).strip() != ''
                if has_actions and has_text:
                    # Parse actions as a dict and check that actions["post_update"] IS set to true, otherwise throw error
                    actions_dict = json.loads(row_dict['actions'])
                    if not actions_dict['post_update'] and not actions_dict['quote']:
                        raise ValueError(f"Row has both text and actions and not post_update or quote: {row_dict}")
                if not (has_actions or has_text):
                    raise ValueError(f"Row has neither text nor actions: {row_dict}")
                if has_actions:
                    msg['actions'] = row_dict['actions']
                else:
                    msg['text'] = row_dict['scrubbed_output']
                
                current_chain.append(msg)
                last_chain_id = chain_id
                
        # Write the last chain
        if current_chain:
            fout.write(json.dumps(current_chain) + '\n')

if __name__ == "__main__":
    process_and_write()