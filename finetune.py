import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Configurações
MODEL_NAME = "Qwen/Qwen3-0.6B"
DATASET_FILE = "dataset.jsonl"
OUTPUT_DIR = "./qwen3-lora-adapter"
EPOCHS = 3
BATCH_SIZE = 1
LEARNING_RATE = 2e-4

# Carregar modelo e tokenizer
print("Carregando modelo base...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cuda",
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Configuração LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Carregar dataset
dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

# Configuração do treinamento
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    logging_steps=1,
    save_strategy="epoch",
    fp16=False,
    gradient_accumulation_steps=4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    use_cpu=False
)

# Treinar
print("Iniciando treinamento...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()

# Salvar adapter
print(f"Salvando adapter em {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuning concluído!")
