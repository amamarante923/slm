import os
import re

from transformers import pipeline, Pipeline

DIRETORIO_ADAPTADOR: str = "./qwen3-lora-adapter"
NOME_MODELO: str = "Qwen/Qwen3-0.6B"
MAX_NOVOS_TOKENS: int = 2000


def carregar_pipeline_com_adaptador(
    nome_modelo: str, diretorio_adaptador: str
) -> Pipeline:
    """Carrega o pipeline com adapter LoRA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizador = AutoTokenizer.from_pretrained(diretorio_adaptador, trust_remote_code=True)
    modelo_base = AutoModelForCausalLM.from_pretrained(
        nome_modelo,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    modelo = PeftModel.from_pretrained(modelo_base, diretorio_adaptador)
    modelo.eval()
    return pipeline("text-generation", model=modelo, tokenizer=tokenizador, max_new_tokens=MAX_NOVOS_TOKENS)


def carregar_pipeline_base(nome_modelo: str) -> Pipeline:
    """Carrega o pipeline com o modelo base."""
    return pipeline("text-generation", model=nome_modelo, max_new_tokens=MAX_NOVOS_TOKENS)


def carregar_pipeline(nome_modelo: str, diretorio_adaptador: str) -> Pipeline:
    """Seleciona e carrega o pipeline adequado."""
    if os.path.exists(diretorio_adaptador):
        print("Adapter LoRA encontrado! Carregando modelo personalizado...")
        return carregar_pipeline_com_adaptador(nome_modelo, diretorio_adaptador)
    print("Usando modelo base...")
    return carregar_pipeline_base(nome_modelo)


def limpar_resposta(texto: str) -> str:
    """Remove artefatos de geração da resposta."""
    return re.sub(r"ground.*?ground", "", texto, flags=re.DOTALL)


def extrair_resposta(resultado: list) -> str:
    """Extrai o conteúdo da resposta gerada pelo modelo."""
    return resultado[0]["generated_text"][1]["content"]


def executar_chat(pipe: Pipeline) -> None:
    """Loop principal de interação com o usuário."""
    while True:
        entrada = input("Enter your prompt (or type 'exit' to quit): ")
        if entrada.lower() == "exit":
            break
        mensagens = [{"role": "user", "content": entrada}]
        resultado = pipe(mensagens)
        resposta = limpar_resposta(extrair_resposta(resultado))
        print(resposta)


if __name__ == "__main__":
    pipe = carregar_pipeline(NOME_MODELO, DIRETORIO_ADAPTADOR)
    executar_chat(pipe)
