from typing import List, Dict, TypedDict


class Message(TypedDict):
    text: str
    user_id: str


Chain = List[Message]  # List of Message dictionaries

SYSTEM_PROMPT = """Complete your message."""

def get_subject_user_id(chain: Chain) -> str:
    """Get the user_id of the last person"""
    return chain[-1]["user_id"]


def create_user_id_mapping(chain: Chain) -> Dict[str, str]:
    """Create a mapping from original user IDs to encoded IDs.
    The last user is encoded as 'assistant', and if their ID appears earlier in the chain,
    it is also encoded as 'assistant'. Other users are encoded as user_1, user_2, user_3, etc."""
    if not chain:
        return {}

    # Get the last user's ID
    last_user_id = get_subject_user_id(chain)

    # Track seen IDs and their encodings
    seen_ids = {last_user_id: "assistant"}
    next_number = 1

    # Process all messages to build the mapping
    for msg in chain:
        user_id = msg["user_id"]
        if user_id not in seen_ids:
            seen_ids[user_id] = f"user_{next_number}"
            next_number += 1

    return seen_ids


def convert_chain_to_text(chain: Chain) -> str:
    """Convert a chain of messages to a single text string with XML-like tags."""
    if not chain:
        return ""

    # Get the user ID mapping
    user_id_mapping = create_user_id_mapping(chain)

    # Process messages using the mapping
    message = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"""
    for msg in chain:
        user_tag = user_id_mapping[msg["user_id"]]
        message += f"<|start_header_id|>{user_tag}<|end_header_id|>\n\n{msg['text']}<|eot_id|>"
    return message
