from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from numpy.typing import NDArray
import numpy as np
import pickle
import logging
from sklearn.model_selection import train_test_split
import os
import datetime
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOME = os.environ.get("HOME", "/home/ubuntu")

# Shared constants
USERNAME = "jchooi"


@dataclass
class Message:
    user_id: str
    relative_integer_time: int
    actions: Optional[str] = None
    text: Optional[str] = None


class ConversationThread:
    messages: list[Message]
    last_user_id: str
    user_id_mapping: Dict[str, str]

    def __init__(self, messages: list[Message]):
        self.messages = messages
        self.last_user_id = self.messages[-1].user_id
        self.user_id_mapping = self._create_user_id_mapping()

    def is_textual(self) -> bool:
        return self.messages[-1].text is not None

    def _create_user_id_mapping(self) -> Dict[str, str]:
        """Create a mapping from original user IDs to encoded IDs.
        The last user is encoded as 'assistant', and if their ID appears earlier in the chain,
        it is also encoded as 'assistant'. Other users are encoded as user_1, user_2, user_3, etc."""
        # Track seen IDs and their encodings
        seen_ids = {self.last_user_id: "assistant"}

        # Process all messages to build the mapping
        next_number = 1
        for message in self.messages:
            user_id = message.user_id
            if not seen_ids.get(user_id):
                seen_ids[user_id] = f"user_{next_number}"
                next_number += 1

        return seen_ids

    def get_llama_text(self) -> Tuple[str, str]:
        """Returns the thread formatted for Llama and split into two.

        The second part is the text of the last user, and the first part is whatever that comes before it.
        """
        SYSTEM_PROMPT = """You are a user on social media. Your goal is to write posts and interact with other users' posts."""

        # Iterate until the very last message
        preceding_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"""
        for message in self.messages[:-1]:
            user_tag = self.user_id_mapping[message.user_id]
            preceding_text += f"<|start_header_id|>{user_tag}<|end_header_id|>\n\n{message.text}<|eot_id|>"
        preceding_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        # answer and eot
        answer_text = f"{self.messages[-1].text}<|eot_id|>"
        return (preceding_text, answer_text)


def ConversationThreadFactory(thread_dict: list[Dict[str, Any]]) -> ConversationThread:
    messages: list[Message] = []
    for message_dict in thread_dict:
        message = Message(
            message_dict["user_id"], message_dict["relative_integer_time"]
        )
        if message_dict.get("actions"):
            message.actions = message_dict["actions"]
        else:
            message.text = message_dict["text"]
        messages.append(message)
    return ConversationThread(messages)


class ConversationDataset(Dataset):
    def __init__(
        self,
        folder_path: Path,
        tokenizer: Any,
        split: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        debug: bool = False # load 1% of the dataset
    ):
        self.tokenizer = tokenizer
        self.examples: list[
            Tuple[str, str, NDArray[np.float32]]
        ] = []  # (full_text, assistant_text, embeddings)

        file_paths = list(folder_path.glob("*.jsonl"))
        embedding_file_path = folder_path / "embeddings.pkl"
        with open(embedding_file_path, "rb") as f:
            cluster_embeddings: list[NDArray[np.float32]] = pickle.load(f)

        # load the clusters
        if debug:
            # load only 5 clusters
            file_paths = file_paths[:5]
        for i, file_path in tqdm(
            enumerate(file_paths), total=len(file_paths), desc="Loading clusters"
        ):
            cluster_embedding: NDArray[np.float32] = cluster_embeddings[i]
            with open(file_path, "r") as f:
                for line in f:
                    chain = json.loads(line)
                    thread = ConversationThreadFactory(chain)
                    if not thread.is_textual():
                        # skip messages that are not textual
                        continue
                    preceding_text, assistant_text = thread.get_llama_text()

                    payload = (preceding_text, assistant_text, cluster_embedding)
                    self.examples.append(payload)

        # Perform train/test split if requested
        if split is not None:
            assert split in ["train", "test"], "split must be either 'train' or 'test'"

            train_examples, test_examples = train_test_split(
                self.examples, test_size=test_size, random_state=random_state
            )
            self.examples = train_examples if split == "train" else test_examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        preceding_text, assistant_text, embeddings = self.examples[idx]
        embeddings = torch.tensor(embeddings)

        # Tokenize the full text for input
        preceding_tokens: Dict[str, Any] = self.tokenizer(
            preceding_text,
            return_tensors="pt",
        )
        assistant_tokens: Dict[str, Any] = self.tokenizer(
            assistant_text,
            return_tensors="pt",
        )

        full_tokens: Dict[str, Any] = self.tokenizer(
            preceding_text + assistant_text,
            return_tensors="pt",
        )

        # Only compute loss on assistant's response
        num_preceding_tokens = len(preceding_tokens["input_ids"])
        assistant_labels: torch.Tensor = assistant_tokens["input_ids"].clone()
        masked_preceding_labels = (
            torch.ones(size=(1, num_preceding_tokens), dtype=assistant_labels.dtype)
            * -100
        )
        labels = torch.concat([masked_preceding_labels, assistant_labels], dim=1)

        payload = {
            "input_ids": full_tokens["input_ids"].squeeze(0),
            "attention_mask": full_tokens["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "text": preceding_text,  # Store original text for reference
        }

        # Dump payload into logs/test.jsonl by appending
        # comment this out if not debugging
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"logs/test_{timestamp}.jsonl"
        with open(log_filename, "w") as f:
            # Convert tensors to lists for JSON serialization
            json_payload = {
                "input_ids": payload["input_ids"].tolist(),
                "attention_mask": payload["attention_mask"].tolist(),
                "labels": payload["labels"].tolist(),
                "text": payload["text"],
            }
            json.dump(json_payload, f)
            f.write("\n")

        return payload


def collate_fn(batch, max_len=2048):
    """Custom collate function to pad/truncate each sequence to the batch max length, capped at 2048."""
    # Find the max length in this batch, but cap at max_len
    batch_max_len = min(max(item["input_ids"].size(-1) for item in batch), max_len)

    def pad_or_truncate(tensor, length):
        if tensor.size(-1) > length:
            return tensor[..., :length]
        elif tensor.size(-1) < length:
            pad_size = [*tensor.shape[:-1], length - tensor.size(-1)]
            return torch.cat(
                [tensor, torch.zeros(*pad_size, dtype=tensor.dtype)], dim=-1
            )
        return tensor

    input_ids = torch.stack(
        [pad_or_truncate(item["input_ids"], batch_max_len) for item in batch]
    )
    attention_mask = torch.stack(
        [pad_or_truncate(item["attention_mask"], batch_max_len) for item in batch]
    )
    labels = torch.stack(
        [pad_or_truncate(item["labels"], batch_max_len) for item in batch]
    )

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_model_and_tokenizer(model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
    """Load base model and tokenizer with common settings."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"No pad tokens. Set {tokenizer.pad_token=}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    return model, tokenizer
