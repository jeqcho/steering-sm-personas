import json
import os
from presidio_anonymizer import BatchAnonymizerEngine
from tqdm import tqdm
import time

import pandas as pd
from presidio_analyzer import (
    AnalyzerEngine,
    PatternRecognizer,
    Pattern,
    BatchAnalyzerEngine,
)
from torch.utils.data import Dataset
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
    logger.info("Starting batch Presidio analysis...")
    df2 = df[["full_text"]]
    df_dict = df2.to_dict(orient="list")
    batch_analyzer = BatchAnalyzerEngine(analyzer_engine=presidio_analyzer)

    # Use the best configuration for the final result
    start_time = time.time()
    analyzer_results = list(
        batch_analyzer.analyze_dict(
            df_dict,
            language="en",
            batch_size=12,
            n_process=8
        )
    )
    # remove irrelevant PII
    # TODO
    
    df["presidio_batch_output"] = analyzer_results[0].recognizer_results
    elapsed = int(time.time() - start_time)
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(
        f"Finished batch Presidio analysis with best config in {hours}h:{minutes}m:{seconds}s"
    )

    # anonymize
    batch_anonymizer = BatchAnonymizerEngine()
    anonymizer_results = batch_anonymizer.anonymize_dict(analyzer_results)
    df["scrubbed_output"] = anonymizer_results["full_text"]

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
