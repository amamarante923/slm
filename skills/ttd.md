---
name: tdd
description: Desenvolvimento guiado por testes (Test-Driven Development)
---

# Objetivo

Implementar funcionalidades seguindo rigorosamente o ciclo TDD:

1. Red
2. Green
3. Refactor

Nunca implementar código de produção antes de existir um teste falhando.

# Processo

## Fase 1 - Red

Antes de modificar qualquer código:

- Escreva um ou mais testes que descrevam o comportamento desejado.
- Execute os testes.
- Confirme que falham pelo motivo esperado.
- Documente a falha.

Checklist:

- [ ] Teste criado
- [ ] Teste executado
- [ ] Falha confirmada

## Fase 2 - Green

Implemente a menor quantidade possível de código para:

- Fazer o teste passar.
- Evitar otimizações prematuras.
- Evitar abstrações desnecessárias.

Checklist:

- [ ] Teste original passou
- [ ] Nenhum comportamento adicional foi implementado

## Fase 3 - Refactor

Após todos os testes passarem:

- Melhorar legibilidade.
- Remover duplicações.
- Aplicar padrões quando fizer sentido.
- Garantir que todos os testes continuem passando.

Checklist:

- [ ] Refatoração concluída
- [ ] Testes executados novamente
- [ ] Todos os testes passaram

# Regras

## Proibido

- Escrever funcionalidade sem teste.
- Criar múltiplas funcionalidades em um único ciclo.
- Refatorar enquanto existem testes falhando.
- Ignorar testes quebrados.

## Obrigatório

- Executar testes frequentemente.
- Trabalhar em pequenas iterações.
- Explicar em qual fase do TDD está.
- Encerrar cada tarefa informando:
  - Testes criados
  - Testes executados
  - Resultado final

# Estrutura de Resposta

Sempre responder usando:

## Fase Atual

Red | Green | Refactor

## Plano

Descrição da ação atual.

## Execução

Passos realizados.

## Resultado

Estado dos testes e da implementação.

# Critérios de Qualidade

O agente deve buscar:

- Alta cobertura dos comportamentos críticos.
- Testes independentes.
- Testes determinísticos.
- Nomes descritivos.
- Código simples.

# Estratégia

Ao receber uma solicitação de implementação:

1. Identificar requisitos.
2. Criar teste mínimo.
3. Executar teste.
4. Implementar solução mínima.
5. Executar testes.
6. Refatorar.
7. Executar testes novamente.
8. Finalizar.

Se o usuário pedir para pular testes, explique os riscos e solicite confirmação explícita antes de prosseguir.