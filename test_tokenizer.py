from transformers import AutoTokenizer
import os
import time
import json

def main():
    try:
        # Initialize tokenizer with local_files_only=True
        print("\nInitializing tokenizer...")
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(
            'Qwen/Qwen2.5-3B-Instruct',
            trust_remote_code=True,
            local_files_only=True  # This will force it to use only local files
        )
        end_time = time.time()
        
        print(f"Tokenizer initialization took {end_time - start_time:.2f} seconds")
        
        # Test the tokenizer
        test_text = "Hello, how are you today?"
        tokens = tokenizer(test_text)
        
        print("\nInput text:", test_text)
        print("Token IDs:", tokens['input_ids'])
        print("Decoded tokens:", tokenizer.decode(tokens['input_ids']))
        
        # Save tokenizer information to a file
        print("\nSaving tokenizer information to file...")
        with open('tokenizer_info.txt', 'w', encoding='utf-8') as f:
            # Write basic tokenizer info
            f.write("Tokenizer Basic Information:\n")
            f.write("=" * 50 + "\n")
            f.write(f"Vocabulary size: {tokenizer.vocab_size}\n")
            f.write(f"Model max length: {tokenizer.model_max_length}\n")
            f.write("\n")
            
            # Write special tokens
            f.write("Special Tokens:\n")
            f.write("=" * 50 + "\n")
            f.write(f"bos_token: {tokenizer.bos_token}\n")
            f.write(f"eos_token: {tokenizer.eos_token}\n")
            f.write(f"pad_token: {tokenizer.pad_token}\n")
            f.write(f"unk_token: {tokenizer.unk_token}\n")
            f.write("\n")
            
            # Write complete vocabulary
            f.write("Complete Vocabulary:\n")
            f.write("=" * 50 + "\n")
            vocab = tokenizer.get_vocab()
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])  # Sort by token ID
            for token, id in sorted_vocab:
                f.write(f"{token}: {id}\n")
            
            # Write tokenizer configuration
            f.write("\nTokenizer Configuration:\n")
            f.write("=" * 50 + "\n")
            config = {
                "model_max_length": tokenizer.model_max_length,
                "padding_side": tokenizer.padding_side,
                "truncation_side": tokenizer.truncation_side,
                "clean_up_tokenization_spaces": tokenizer.clean_up_tokenization_spaces,
                "split_special_tokens": tokenizer.split_special_tokens,
            }
            f.write(json.dumps(config, indent=2))
        
        print("Tokenizer information saved to 'tokenizer_info.txt'")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("\nIf you see a 'file not found' error, it means the tokenizer files are not in the cache.")
        print("You'll need to download them first on a machine with internet access and then copy them to this machine.")

if __name__ == "__main__":
    main() 