# Identificação ARX e Controle Ótimo Discreto para Alocação em ITSA4

Identificação de sistemas (modelo ARX estimado por regressão ridge) combinada
com controle ótimo discreto para alocação dinâmica semanal em ITSA4 (Itaúsa).
A cada semana decide-se a fração `u[k] ∈ [0, 1]` da carteira em ITSA4; a
parcela restante é remunerada pelo CDI.

O projeto compara três leis de controle sob o mesmo simulador — um controlador
de horizonte 1 (solução analítica do custo quadrático), um controle preditivo
(MPC) de horizonte deslizante com restrição de caixa e um regulador de
volatilidade — e analisa o sistema e o controlador com as ferramentas da
disciplina:

- Função de transferência `G(z) = B(z)/A(z)` do ARX: polos, estabilidade
  (`|z_i| < 1`), resposta em frequência e sensibilidade dos polos à
  regularização do ridge.
- O controlador H=1 e o estimador EWMA de volatilidade caracterizados como
  filtros discretos de 1ª ordem `H(z) = b z/(z - a)`: polo, ganho DC,
  constante de tempo e banda passante de −3 dB.
- Seleção de hiperparâmetros (λ, ρ) pela metodologia do tutorial de otimização
  de CMC-12: busca em grade seguida de refino por Nelder-Mead (`fminsearch`),
  com o ótimo da grade como chute inicial.
- Experimento de atraso de transporte: degradação do desempenho quando a
  previsão chega com 1–2 semanas extras de atraso.
- Avaliação estatística: Sharpe de excesso sobre o CDI, bootstrap de blocos,
  validação walk-forward com re-identificação (janelas de seleção e de
  avaliação disjuntas) e sensibilidade ao custo de transação.

## Arquitetura

```text
Exame/
├── main.py                 # orquestrador (fluxo de ponta a ponta)
├── requirements.txt        # dependências com versões fixadas
├── src/
│   ├── config.py           # configuração única e imutável
│   ├── dados.py            # download (Yahoo/BCB), snapshot, séries semanais
│   ├── features.py         # regressores do ARX + volatilidade EWMA causal
│   ├── identificacao.py    # ARX por ridge com pesos temporais
│   ├── estrategias.py      # ControleH1, ControleMPC(WF), VolTarget, BuyAndHold
│   ├── backtest.py         # simulador único, grade + Nelder-Mead, atraso, walk-forward
│   ├── metricas.py         # métricas financeiras e significância (bootstrap)
│   ├── funcao_transferencia.py  # G(z), polos, filtros de 1ª ordem
│   └── relatorio.py        # geração de CSVs e figuras
├── tests/
│   └── test_projeto.py     # suíte de verificação (offline, sem internet)
├── outputs/                # tabelas (.csv) e snapshot dos dados
└── figuras/                # figuras (.png) usadas pelo relatório
```

Cada estratégia apenas decide `u[k]`; um único `backtest.simular` aplica a
mesma dinâmica de capital a todas, garantindo comparação justa.

## Separação temporal (sem sobreposição seleção/avaliação)

Treino 2022–2023 → validação 2024 (escolha de λ, ρ, H) → re-treino até
2025-12-20 → teste 2026. O walk-forward (2025 em diante) re-identifica o
modelo semanalmente só com o passado e usa hiperparâmetros escolhidos em 2024,
permanecendo integralmente fora da amostra.

## Como executar

Na pasta do projeto, com um ambiente virtual ativo:

```bat
pip install -r requirements.txt
py tests\test_projeto.py     # verificação offline (rápida, sem internet)
py main.py                   # pipeline completo (baixa dados na 1ª execução)
```

Na primeira execução os dados são baixados (Yahoo Finance e SGS/BCB) e salvos
em `outputs/snapshot/` para reprodutibilidade; execuções seguintes recarregam
do snapshot. Para forçar novo download, apague `outputs/snapshot/`.
