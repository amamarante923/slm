import os
import re

ADAPTER_DIR = "./qwen3-lora-adapter"
MODEL_NAME = "Qwen/Qwen3-0.6B"

if os.path.exists(ADAPTER_DIR):
    print("Adapter LoRA encontrado! Carregando modelo personalizado...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    from transformers import pipeline
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=2000)
else:
    print("Usando modelo base...")
    from transformers import pipeline
    pipe = pipeline('text-generation', model=MODEL_NAME, max_new_tokens=2000)

while True:
    start = input("Enter your prompt (or type 'exit' to quit): ")
    if start.lower() == 'exit':
        break
    messages = [
        {"role": "user", "content": start},
    ]
    result = pipe(messages)[0]['generated_text'][1]['content']
    result = re.sub(r'ground.*?ground', '', result, flags=re.DOTALL)
    print(result)
