import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset

NOME_MODELO: str = "Qwen/Qwen3-0.6B"
ARQUIVO_DATASET: str = "dataset.jsonl"
DIRETORIO_SAIDA: str = "./qwen3-lora-adapter"
EPOCAS: int = 3
TAMANHO_LOTE: int = 1
TAXA_APRENDIZADO: float = 2e-4


def carregar_tokenizador(nome_modelo: str) -> PreTrainedTokenizerBase:
    """Carrega e configura o tokenizador."""
    tokenizador = AutoTokenizer.from_pretrained(nome_modelo, trust_remote_code=True)
    if tokenizador.pad_token is None:
        tokenizador.pad_token = tokenizador.eos_token
    return tokenizador


def carregar_modelo_base(nome_modelo: str) -> PreTrainedModel:
    """Carrega o modelo base para fine-tuning."""
    return AutoModelForCausalLM.from_pretrained(
        nome_modelo,
        torch_dtype=torch.float32,
        device_map="cuda",
        trust_remote_code=True,
    )


def aplicar_lora(modelo: PreTrainedModel) -> PeftModel:
    """Aplica configuração LoRA ao modelo."""
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(modelo, config)


def carregar_dataset(arquivo: str) -> Dataset:
    """Carrega o dataset de treinamento."""
    return load_dataset("json", data_files=arquivo, split="train")


def criar_argumentos_treino(diretorio_saida: str, epocas: int, tamanho_lote: int, taxa_aprendizado: float) -> SFTConfig:
    """Cria a configuração de treinamento."""
    return SFTConfig(
        output_dir=diretorio_saida,
        num_train_epochs=epocas,
        per_device_train_batch_size=tamanho_lote,
        learning_rate=taxa_aprendizado,
        logging_steps=1,
        save_strategy="epoch",
        fp16=False,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        use_cpu=False,
    )


def treinar(modelo: PeftModel, dataset: Dataset, argumentos: SFTConfig) -> None:
    """Executa o treinamento supervisionado."""
    treinador = SFTTrainer(model=modelo, train_dataset=dataset, args=argumentos)
    treinador.train()


def salvar_modelo(modelo: PeftModel, tokenizador: PreTrainedTokenizerBase, diretorio: str) -> None:
    """Salva o adapter e tokenizador treinados."""
    modelo.save_pretrained(diretorio)
    tokenizador.save_pretrained(diretorio)


def executar_finetune() -> None:
    """Pipeline completo de fine-tuning."""
    print("Carregando modelo base...")
    tokenizador = carregar_tokenizador(NOME_MODELO)
    modelo_base = carregar_modelo_base(NOME_MODELO)
    modelo = aplicar_lora(modelo_base)
    modelo.print_trainable_parameters()

    dataset = carregar_dataset(ARQUIVO_DATASET)
    argumentos = criar_argumentos_treino(DIRETORIO_SAIDA, EPOCAS, TAMANHO_LOTE, TAXA_APRENDIZADO)

    print("Iniciando treinamento...")
    treinar(modelo, dataset, argumentos)

    print(f"Salvando adapter em {DIRETORIO_SAIDA}...")
    salvar_modelo(modelo, tokenizador, DIRETORIO_SAIDA)
    print("Fine-tuning concluído!")


if __name__ == "__main__":
    executar_finetune()
