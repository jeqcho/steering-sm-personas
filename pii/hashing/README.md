Note: This folder is copied from [Scezaquer/SM-based-personas](https://github.com/Scezaquer/SM-based-personas/tree/master/data_processing/hashing)

IMPORTANT: DO NOT EXPOSE secret_hash_did.tok TO THE INTERNET. ADD IT TO YOUR
GITIGNORE. DO NOT PUSH IT TO THE REPO. DO NOT SHARE IT IN AN UNENCRYPTED OR
UNSAFE MANNER. ONLY SHARE IT WITH TRUSTED PARTIES.

## How Hashing Works

The hashing process ensures that user DIDs are anonymized while maintaining their connection within a specific message chain.

1.  **Secret Key:** A secret key is read from `secret_hash_did.tok`.
2.  **Chain Context:** For each message chain, the `user_id` field is temporarily removed from all messages.
3.  **JSON Dump:** The remaining message chain (without `user_id`s) is converted into a sorted JSON string (dump).
4.  **Hashing:** Each original `user_id` is concatenated with the JSON dump and the secret key (`f"{did}{dump}{SECRET}"`).
5.  **SHA-256:** The resulting string is hashed using SHA-256.
6.  **Replacement:** The original `user_id` in each message is replaced with its corresponding hash.

This method ensures that the same DID will have a different hash if it appears in different message chains, adding a layer of contextual security.

The `hash_did.py` script applies this process to the dataset, creating hashed versions of the data files and a mapping of hashed DIDs to cluster IDs.

The `is_did_in_chain.py` script uses the same hashing logic to check if a given raw DID belongs to a specific chain by comparing its calculated hash against the hashes present in the chain.

