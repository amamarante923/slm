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

## Observações

- O código remove trechos entre `<think>` e `</think>`, o que pode ser útil para ocultar pensamentos internos gerados pelo modelo.
