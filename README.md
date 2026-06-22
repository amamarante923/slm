# SLM

Este repositório contém um script Python simples chamado `slm.py` que utiliza a biblioteca `transformers` para gerar texto a partir de um modelo de linguagem pequeno (apenas 600 milhões de parâmetros) e local - um SLM (Small Language Model).

## O que o script faz

O `slm.py`:

- configura um pipeline de geração de texto usando o modelo `Qwen/Qwen3-0.6B`;
- entra em um loop interativo onde o usuário pode digitar um prompt;
- envia o prompt para o modelo e obtém o texto gerado;
- remove qualquer conteúdo entre as tags `<think></think>` do resultado;
- exibe o texto final para o usuário;
- encerra o programa quando o usuário digita `exit`.
## Requisitos

- **Python 3.12** (versões mais recentes podem não ser compatíveis com todas as dependências).
## Como usar

1. Crie e ative a venv:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1 
```

2. Instale as dependências necessárias:

```bash
pip install -r requirements.txt
```

3. Execute o script:

```bash
python slm.py
```

4. Digite o prompt desejado e pressione Enter.
5. Para sair, digite `exit`.

## Treinamento (fine-tuning)

O script `finetune.py` permite ajustar o modelo `Qwen3-0.6B` aos seus dados usando **LoRA** (Low-Rank Adaptation), uma técnica leve de fine-tuning que treina apenas uma pequena fração dos parâmetros do modelo.

### Como funciona

- Carrega o modelo base `Qwen/Qwen3-0.6B`;
- Aplica adaptadores LoRA nas camadas de atenção (`q_proj`, `v_proj`);
- Treina com o `SFTTrainer` da biblioteca `trl` usando dados no formato conversacional;
- Salva o adaptador treinado na pasta `./qwen3-lora-adapter`.

### Formato do dataset

Crie um arquivo `dataset.jsonl` na raiz do projeto. Cada linha deve ser um objeto JSON com o seguinte formato:

```json
{"messages": [{"role": "user", "content": "Sua pergunta aqui"}, {"role": "assistant", "content": "Resposta desejada aqui"}]}
```

### Como treinar

```bash
python finetune.py
```

Após o treinamento, o `slm.py` detecta automaticamente a pasta `./qwen3-lora-adapter` e carrega o adaptador ao iniciar.

## Observações

- O código remove trechos entre `<think>` e `</think>`, o que pode ser útil para ocultar pensamentos internos gerados pelo modelo.
