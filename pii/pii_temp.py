import re
import json
import os
from typing import Any, Dict
from tqdm import tqdm
import time

import pandas as pd
import numpy as np
import torch
from presidio_analyzer import (
    AnalyzerEngine,
    PatternRecognizer,
    Pattern,
    BatchAnalyzerEngine,
)
from torch.utils.data import Dataset
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from loguru import logger

tqdm.pandas()

##############################

##############################
# 2) general pii detection + custom regex


def presidio_analyze(text, analyzer, allow_list, entities_excluded):
    result_list = analyzer.analyze(text, language="en", allow_list=allow_list)
    excluded_set = set(entities_excluded)
    return [result for result in result_list if result.entity_type not in excluded_set]


##############################
# 3 - NER
class pii_dataset(Dataset):
    """
    PII dataset.
    """

    def __init__(self, data: pd.DataFrame):
        """
        PII Dataset

        Parameters:
        -----------
        data: pd.DataFrame
            The dataset. It must contain the following columns:
            ["full_text"]
        """
        self.data = data

    def __len__(self) -> int:
        """Size of our dataset"""
        return len(self.data)

    def __getitem__(self, item):
        """One row in our dataset"""
        if item >= len(self):
            raise IndexError
        return self.data["full_text"][item]


#################
# 5) final labels and performance
def set_final_labels(df, output_col_list):
    df["final_flag"] = (
        ~df[output_col_list].isnull() & df[output_col_list].astype(bool)
    ).any(axis=1)
    return df


if __name__ == "__main__":
    logger.info("Starting PII detection process...")

    # Load Configs
    logger.info("Loading configuration...")
    with open("conf.json", "r") as config:
        params = json.load(config)

    # Read File
    logger.info("Reading and processing input file...")

    # Read parquet file
    df = pd.read_parquet(params["file_name"])

    # Create a working DataFrame with only non-empty text entries
    df["full_text"] = df["text"]

    batch_size = 256

    # 2.0 Setup Presidio
    logger.info("Setting up Presidio analyzer...")
    presidio_analyzer = AnalyzerEngine()

    for entity_dict in tqdm(
        params["presidio_deny_list"], desc="Adding Presidio recognizers"
    ):
        if "pattern" in entity_dict:
            pattern_recognizer = PatternRecognizer(
                supported_entity=entity_dict["entity"],
                deny_list=entity_dict["deny_list"],
                patterns=[
                    Pattern(
                        name=entity_dict["entity"],
                        regex=entity_dict["pattern"],
                        score=1,
                    )
                ],
            )
        else:
            pattern_recognizer = PatternRecognizer(
                supported_entity=entity_dict["entity"],
                deny_list=entity_dict["deny_list"],
            )

        presidio_analyzer.registry.add_recognizer(pattern_recognizer)

    entities_excluded = params["presidio_exclusion_list"]

    # 2.1 Run Presidio
    # logger.info("Running Presidio analysis...")
    # df["presidio_output"] = df["full_text"].progress_apply(
    #     presidio_analyze,
    #     analyzer=presidio_analyzer,
    #     allow_list=params["presidio_allow_list"],
    #     entities_excluded=params["presidio_exclusion_list"],
    # )
    # logger.info("Finished presidio analysis")

    # alt presidio
    logger.info("Starting batch Presidio analysis with grid search...")
    df2 = df[["full_text"]]
    df_dict = df2.to_dict(orient="list")
    batch_analyzer = BatchAnalyzerEngine(analyzer_engine=presidio_analyzer)
    
    # Grid search parameters
    batch_sizes = [6, 8, 10, 12]
    n_processes = [6, 8, 10]
    
    # Table to store results
    results = []
    
    logger.info("Batch size | N processes | Time (seconds)")
    logger.info("---------------------------------------")
    
    for batch_size in batch_sizes:
        for n_process in n_processes:
            start_time = time.time()
            analyzer_results = list(batch_analyzer.analyze_dict(
                df_dict, language="en", batch_size=batch_size, n_process=n_process
            ))
            elapsed_time = time.time() - start_time
            results.append((batch_size, n_process, elapsed_time))
            logger.info(f"{batch_size:^10} | {n_process:^11} | {elapsed_time:.2f}")
    
    # Find the best configuration
    best_config = min(results, key=lambda x: x[2])
    logger.info(f"Best configuration: batch_size={best_config[0]}, n_process={best_config[1]}, time={best_config[2]:.2f}s")
    
    # Use the best configuration for the final result
    start_time = time.time()
    analyzer_results = list(batch_analyzer.analyze_dict(
        df_dict, language="en", batch_size=best_config[0], n_process=best_config[1]
    ))
    df["presidio_batch_output"] = analyzer_results[0].recognizer_results
    logger.info(f"Finished batch Presidio analysis with best config in {time.time() - start_time:.2f} seconds")

    # 5 - Calculate output & score
    logger.info("Calculating final labels...")
    output_col_list = ["presidio_batch_output"]
    df = set_final_labels(df, output_col_list)

    # 6 - Export
    logger.info("Exporting results...")
    directory = params["experiment_name"]
    if not os.path.exists(directory):
        os.mkdir(directory)

    df.to_csv(directory + "/pii_dataset_tags.csv", index=False)
    logger.info("Process completed successfully!")
