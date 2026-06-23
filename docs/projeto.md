# Documentação do Projeto SLM

## Visão Geral

Projeto para inferência e fine-tuning de modelos de linguagem pequenos (SLM - Small Language Models) utilizando o modelo **Qwen/Qwen3-0.6B** com adaptadores LoRA.

## Módulos

### slm.py — Inferência

Pipeline de chat interativo com suporte a adaptadores LoRA.

**Constantes:**
- `NOME_MODELO` — Modelo base utilizado (`Qwen/Qwen3-0.6B`)
- `DIRETORIO_ADAPTADOR` — Caminho do adaptador LoRA (`./qwen3-lora-adapter`)
- `MAX_NOVOS_TOKENS` — Limite de tokens gerados (`2000`)

**Funções:**
- `carregar_pipeline_com_adaptador(nome_modelo, diretorio_adaptador)` — Carrega pipeline com adaptador LoRA treinado
- `carregar_pipeline_base(nome_modelo)` — Carrega pipeline com modelo base sem adaptador
- `carregar_pipeline(nome_modelo, diretorio_adaptador)` — Detecta automaticamente se o adaptador existe e carrega o pipeline apropriado
- `limpar_resposta(texto)` — Remove artefatos de geração (padrão `ground...ground`)
- `extrair_resposta(resultado)` — Extrai o conteúdo da resposta do assistente
- `executar_chat(pipe)` — Loop interativo de chat no terminal

### finetune.py — Fine-Tuning

Pipeline de fine-tuning supervisionado com LoRA.

**Constantes:**
- `NOME_MODELO` — Modelo base (`Qwen/Qwen3-0.6B`)
- `ARQUIVO_DATASET` — Arquivo de dados (`dataset.jsonl`)
- `DIRETORIO_SAIDA` — Diretório de saída do adaptador (`./qwen3-lora-adapter`)
- `EPOCAS`, `TAMANHO_LOTE`, `TAXA_APRENDIZADO` — Hiperparâmetros

**Funções:**
- `carregar_tokenizador(nome_modelo)` — Carrega e configura tokenizador com pad_token
- `carregar_modelo_base(nome_modelo)` — Carrega modelo para treinamento
- `aplicar_lora(modelo)` — Aplica configuração LoRA (r=8, alpha=16, dropout=0.05)
- `carregar_dataset(arquivo)` — Carrega dataset JSONL
- `criar_argumentos_treino(...)` — Cria configuração SFTConfig
- `treinar(modelo, dataset, argumentos)` — Executa treinamento
- `salvar_modelo(modelo, tokenizador, diretorio)` — Salva adaptador e tokenizador

## Testes

Os testes estão em `tests/` e utilizam `unittest` com mocks para isolar dependências externas (torch, transformers, peft).

```bash
python -m pytest tests/
```

## Dependências

Listadas em `requirements.txt`:
- torch
- transformers
- peft
- trl
- datasets
- accelerate
