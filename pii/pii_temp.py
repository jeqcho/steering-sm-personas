import json
import os
from presidio_analyzer.recognizer_registry.recognizer_registry import RecognizerRegistry
from presidio_analyzer.recognizer_result import RecognizerResult
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


def print_elapsed(start, label="Block"):
    elapsed = time.time() - start
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"{label} took {hours}h:{minutes}m:{seconds}s")


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
    df = df.rename(columns={"text": "full_text"})

    batch_size = params.get("batch_size", 256)
    threshold = params.get("threshold", 0)
    n_process = params.get("n_process", 8)

    # 2.0 Setup Presidio
    logger.info("Setting up Presidio analyzer...")
    registry_all = RecognizerRegistry()
    registry_all.load_predefined_recognizers()
    registry = RecognizerRegistry()
    recognizers = registry_all.get_recognizers(
        language="en", entities=params["presidio_inclusion_list"]
    )
    for recognizer in recognizers:
        registry.add_recognizer(recognizer)

    # srcub @username.blsky.social
    bluesky_pattern = Pattern(
        name="bluesky_pattern", regex="(?<=@)[A-Za-z0-9\\-]+\\.bsky\\.social", score=0.9
    )
    # Define the recognizer with one or more patterns
    bluesky_recognizer = PatternRecognizer(
        supported_entity="USERNAME", patterns=[bluesky_pattern]
    )
    registry.add_recognizer(bluesky_recognizer)

    presidio_analyzer = AnalyzerEngine(registry=registry)

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
    df_dict = df[["full_text"]].to_dict(orient="list")
    batch_analyzer = BatchAnalyzerEngine(analyzer_engine=presidio_analyzer)

    # Use the best configuration for the final result
    start = time.time()
    analyzer_results = list(
        batch_analyzer.analyze_dict(
            df_dict,
            language="en",
            batch_size=batch_size,
            n_process=n_process,
            score_threshold=threshold,
        )
    )
    print_elapsed(start, "Presidio batch analysis")

    # remove @user
    start = time.time()
    new_results: list[list[RecognizerResult]] = []
    lowest_scores: list[float] = []  # Store the lowest score for each row
    blsky_str = ".bsky.social"
    for i, result in enumerate(analyzer_results[0].recognizer_results):
        assert isinstance(result, list)
        new_result: list[RecognizerResult] = []
        for subresult in result:
            if subresult.entity_type == "URL" and subresult.start > 0:
                string = analyzer_results[0].value[i]
                if string[subresult.start - 1] == "@":
                    # username
                    if string[:subresult.end].endswith(blsky_str):
                        # blsky.social username
                        continue
                    else:
                        # custom username
                        subresult.entity_type = "USERNAME"
                        new_result.append(subresult)
            new_result.append(subresult)
        # Compute the lowest score for this new_result, default to 2 if empty
        if new_result:
            min_score = min([sub.score for sub in new_result])
        else:
            min_score = 2
        lowest_scores.append(min_score)
        new_results.append(new_result)
    analyzer_results[0].recognizer_results = new_results
    # Add the lowest score as a new column
    df["lowest_score"] = lowest_scores
    print_elapsed(start, "Presidio removal of unwanted entities")

    df["presidio_batch_output"] = analyzer_results[0].recognizer_results

    # anonymize
    start = time.time()
    batch_anonymizer = BatchAnonymizerEngine()
    anonymizer_results = batch_anonymizer.anonymize_dict(analyzer_results)
    df["scrubbed_output"] = anonymizer_results["full_text"]
    print_elapsed(start, "Presidio scrubbing")

    # 5 - Calculate output & score
    logger.info("Calculating final labels...")
    output_col_list = ["presidio_batch_output"]
    df = set_final_labels(df, output_col_list)

    # 6 - Export
    logger.info("Exporting results...")
    directory = params["experiment_name"]
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    df['presidio_batch_output'] = df['presidio_batch_output'].astype(str)
    
    # for dubigging
    # df.to_csv(directory + "/pii_dataset_tags.csv", index=False)

    df.to_parquet(directory + "/pii_dataset_tags.parquet", index=False)
    logger.info("Process completed successfully!")
