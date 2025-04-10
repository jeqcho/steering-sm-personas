import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def test_qwen_chat_template():
    # Load model and tokenizer
    model_name = 'Qwen/Qwen2.5-3B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto"
    )

    # Create a chat with role information in Qwen's format
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of Belgium?"},
        {"role": "user_2", "content": "What is the capital of UK?"}
    ]

    # Apply chat template
    chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print(f"{chat=}")

    # Tokenize and generate
    inputs = tokenizer(chat, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )

    # Decode and print the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print("\nChat Template:")
    print(chat)
    print("\nModel Response:")
    print(response)
    print("\nModel Response (Raw):")
    print(raw_response)

if __name__ == "__main__":
    test_qwen_chat_template()