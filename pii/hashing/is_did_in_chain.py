import json
import hashlib

with open("secret_hash_did.tok", 'r') as f:
    SECRET = f.read().strip()

def is_did_in_chain(chain, did):
    hashed_dids = []
    for message in chain:
        hashed_dids.append(message.pop("user_id"))
    dump = json.dumps(chain, sort_keys=True)
    for message in chain:
        hashed_did = hashlib.sha256(f"{did}{dump}{SECRET}".encode()).hexdigest()
        if hashed_did in hashed_dids:
            print(f"Found hashed DID: {hashed_did}")
            return True

    return False

example_chain = [{"unix_epoch": 0, "text": "The \ud83c\uddec\ud83c\udde7 govt is about to face a fundamental choice. Does it stand up to help defend the global liberal rules based order that it helped to launch when it co-proclaimed the Atlantic Charter in 1941? Or will it instead try to appease Trump to escape his tariffs and his adolescent wrath? Destiny calls.", "user_id": "f66d3bbe572d66835231c2f887b7fbaa6b2f76e7c1eab4fe6dda4e8d1aac28c4"}]
example_did = "did:plc:lcbuqpy74exvkmrzhbogi6vt"
print(is_did_in_chain(example_chain, example_did))