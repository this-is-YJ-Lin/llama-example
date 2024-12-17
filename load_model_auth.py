from transformers import LlamaForCausalLM, LlamaTokenizer

model_dir = "/flask-test/Llama-3.2-1B/original"
tokenizer = LlamaTokenizer.from_pretrained(model_dir)
model = LlamaForCausalLM.from_pretrained(model_dir)

print("Model and tokenizer loaded successfully.")