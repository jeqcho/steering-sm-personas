from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

AutoTokenizer.from_pretrained(MODEL_NAME)
AutoModel.from_pretrained(MODEL_NAME)