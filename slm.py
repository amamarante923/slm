import warnings
import re
from transformers import pipeline

warnings.filterwarnings('ignore')

pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B", max_new_tokens=2000)

while True:
    start = input("Enter your prompt (or type 'exit' to quit): ")
    if start.lower() == 'exit':
        break
    messages = [
        {"role": "user", "content": start},
    ]
    result = pipe(messages)[0]['generated_text'][1]['content']
    # Remove tudo entre tags <think></think>
    result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
    print(result)