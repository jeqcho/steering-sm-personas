from shared import load_model_and_tokenizer

def main():
    # Load the tokenizer
    _, tokenizer = load_model_and_tokenizer()
    
    print("Tokenizer loaded! Enter strings to see their token IDs (type 'quit' to exit)")
    print("=" * 50)
    
    while True:
        try:
            # Get user input
            text = input("\nEnter text to tokenize: ")
            
            # Check if user wants to quit
            if text.lower() == 'quit':
                break
                
            # Tokenize the input
            tokens = tokenizer.encode(text)
            
            # Print results
            print("\nToken IDs:", tokens)
            print("Number of tokens:", len(tokens))
            
            # Print token to ID mapping
            print("\nToken to ID mapping:")
            for token_id in tokens:
                token = tokenizer.decode([token_id])
                print(f"'{token}' -> {token_id}")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main() 