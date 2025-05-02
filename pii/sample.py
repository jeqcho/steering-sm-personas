import pandas as pd
from pathlib import Path

def main():
    # Read the original parquet file
    input_file = Path.home() / 'all_messages' / 'test_messages.parquet'
    output_file_1 = Path.home() / 'all_messages' / 'subsample_1k.parquet'
    output_file_10 = Path.home() / 'all_messages' / 'subsample_10k.parquet'
    output_file_100 = Path.home() / 'all_messages' / 'subsample_50k.parquet'
    output_file_1_csv = Path.home() / 'all_messages' / 'subsample_1k.csv'
    
    print(f"Reading from {input_file}")
    df = pd.read_parquet(input_file)
    
    # Sample 1000 rows randomly
    sampled_df_1 = df.sample(n=1000, random_state=42)  # random_state for reproducibility
    sampled_df_10 = df.sample(n=10*1000, random_state=42)  # random_state for reproducibility
    sampled_df_100 = df.sample(n=50*1000, random_state=42)  # random_state for reproducibility
    
    # Save the sampled data
    sampled_df_1.to_parquet(output_file_1, index=False)
    sampled_df_10.to_parquet(output_file_10, index=False)
    sampled_df_100.to_parquet(output_file_100, index=False)
    sampled_df_1.to_csv(output_file_1_csv, index=False)

if __name__ == "__main__":
    main() 