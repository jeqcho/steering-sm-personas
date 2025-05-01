import re
import json
import os

import pandas as pd
import numpy as np
import torch
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from torch.utils.data import Dataset
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from loguru import logger


##############################

##############################
# 2) general pii detection + custom regex


def presidio_analyze(text, analyzer, allow_list, entities_excluded):
    result_list = analyzer.analyze(text, language='en', allow_list=allow_list)
    out = []
    while result_list:
        result = result_list.pop()
        if result.entity_type not in entities_excluded:
            out.append(result)
    return out


##############################
# 3 - NER
class pii_dataset(Dataset):
    '''
    PII dataset.
    '''

    def __init__(self, data: pd.DataFrame):
        '''
        PII Dataset

        Parameters:
        -----------
        data: pd.DataFrame
            The dataset. It must contain the following columns:
            ["full_text"]
        '''
        self.data = data

    def __len__(self) -> int:
        ''' Size of our dataset'''
        return len(self.data)

    def __getitem__(self, item):
        '''One row in our dataset'''
        if item >= len(self): raise IndexError
        return self.data["full_text"][item]


def ner_output(dataset, analyzer, entities_excluded, threshold=0.95, batch_size=8):
    '''
    Runs the NER pipeline using the provided dataset.
    '''
    final_output = []
    ner_out = analyzer(dataset, batch_size=batch_size)

    for ner_row in ner_out:
        res = []
        for entity in ner_row:
            if (entity["entity"] not in entities_excluded) and entity["score"] > threshold:
                res.append(entity)
        final_output.append(res)
    return final_output


###########
# 4) NLI

def entity_score(classifier, dataset, entity_list, entity_thresholds=None, batch_size=8):
    '''
    Runs the NLI pipeline using the provided dataset.
    '''
    final_output = []
    # No entity threshold if not provided
    if entity_thresholds is None:
        entity_thresholds = np.zeros(len(entity_list))

    nli_output = classifier(dataset, entity_list, batch_size=batch_size, multi_label=True)

    for nli_row in nli_output:
        res = {}
        for label, score in zip(nli_row["labels"], nli_row["scores"]):
            threshold_index = entity_list.index(label)
            threshold = entity_thresholds[threshold_index]
            if score > threshold:
                res[label] = score
            else:
                res[label] = 0

        final_output.append(res)

    return final_output


#################
# 5) final labels and performance
def set_final_labels(df, output_col_list):
    df["final_flag"] = (~df[output_col_list].isnull() &
                        df[output_col_list].astype(bool)).any(axis=1)
    return df



if __name__ == "__main__":

    # Load Configs
    with open('conf.json', 'r') as config:
        params = json.load(config)

    # Read File
    df = pd.read_csv(params["file_name"])
    df = df[df["full_text"].notnull()].reset_index()
    batch_size = 128

    # 2.0 Setup Presidio
    presidio_analyzer = AnalyzerEngine()

    for entity_dict in params["presidio_deny_list"]:
        if "pattern" in entity_dict:
            pattern_recognizer = PatternRecognizer(supported_entity=entity_dict["entity"],
                                                   deny_list=entity_dict["deny_list"],
                                                   patterns=[Pattern(name=entity_dict["entity"],
                                                                     regex=entity_dict["pattern"],
                                                                     score=1)])
        else:
            pattern_recognizer = PatternRecognizer(supported_entity=entity_dict["entity"],
                                                   deny_list=entity_dict["deny_list"])

        presidio_analyzer.registry.add_recognizer(pattern_recognizer)

    entities_excluded = params["presidio_exclusion_list"]
    entities_excluded = entities_excluded + [term.lower() for term in entities_excluded] + \
                        [term.upper() for term in entities_excluded]

    # 2.1 Run Presidio
    df["presidio_output"] = df["full_text"].apply(presidio_analyze, analyzer=presidio_analyzer,
                                                  allow_list=params["presidio_allow_list"],
                                                  entities_excluded=params["presidio_exclusion_list"])
    logger.info("Finished presidio")

    # 3.0 Run NER Pipeline
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained(
        "dslim/bert-base-NER")

    # Pipeline doesn't take in torch.device but an int...
    device = 0 if torch.cuda.is_available() else -1
    nlp_pipeline = pipeline(task="ner",
                            model=model,
                            tokenizer=tokenizer,
                            # aggregation_strategy="simple",
                            device=device)

    dataset = pii_dataset(df)

    df["ner_output"] = ner_output(dataset=dataset,
                                  analyzer=nlp_pipeline,
                                  entities_excluded=params["ner_exclusion_list"],
                                  batch_size=batch_size
                                  )
    logger.info("Finished NER")
    # 4 - NLI entity

    nli_classifier = pipeline("zero-shot-classification",
                              model="facebook/bart-large-mnli",
                              device=device)

    entities = params["nli_entities"].keys()
    entity_thresholds = params["nli_entities"].values()
    nli_output = entity_score(classifier=nli_classifier,
                              dataset=dataset,
                              entity_list=list(entities),
                              entity_thresholds=list(entity_thresholds),
                              batch_size=batch_size)
    nli_output = np.array(nli_output)

    for entity in entities:
        df[f"nli_{entity}"] = [row[entity] for row in nli_output]

    logger.info("Finished NLI")

    # 5 - Calculate output & score
    output_col_list = ["name_match", "presidio_output", "ner_output"]
    for i, entity in enumerate(params["nli_entities"].keys()):
        output_col_list.append(f"nli_{entity}")
    df = set_final_labels(df, output_col_list)

    # 6 - Export
    directory = params["experiment_name"]
    if not os.path.exists(directory):
        os.mkdir(directory)


    df.to_csv(directory + "/pii_dataset_tags.csv", index=False)
