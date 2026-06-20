# SLM

Este repositório contém um script Python simples chamado `slm.py` que utiliza a biblioteca `transformers` para gerar texto a partir de um modelo de linguagem.

## O que o script faz

O `slm.py`:

- importa avisos e expressões regulares (`warnings` e `re`);
- configura um pipeline de geração de texto usando o modelo `Qwen/Qwen3-0.6B`;
- entra em um loop interativo onde o usuário pode digitar um prompt;
- envia o prompt para o modelo e obtém o texto gerado;
- remove qualquer conteúdo entre as tags `<think></think>` do resultado;
- exibe o texto final para o usuário;
- encerra o programa quando o usuário digita `exit`.

## Como usar

1. Instale as dependências necessárias, por exemplo:

```bash
pip install transformers
```

2. Execute o script:

```bash
python slm.py
```

3. Digite o prompt desejado e pressione Enter.
4. Para sair, digite `exit`.

## Observações

- O modelo utilizado é `Qwen/Qwen3-0.6B` e deve estar disponível no ambiente ou ser baixado automaticamente pelo `transformers`.
- O script suprime avisos para deixar a saída mais limpa.
- O código remove trechos entre `<think>` e `</think>`, o que pode ser útil para ocultar pensamentos internos gerados pelo modelo.
