# Projeto Nexus — Classificação de Risco de Churn (Sprint 3)

Classificação automática de trechos de transcrições de reunião quanto a
risco de churn, usando TF-IDF + SMOTE + Naive Bayes / Logistic Regression
(comparação oficial) e Random Forest (extra).

## Estrutura

```
projeto_nexus_churn/
├── dados/                     # CSVs de entrada (frase + risco_churn)
│   ├── transcricoes_treino.csv
│   └── transcricoes_teste.csv
├── scripts/                   # um script por passo, nessa ordem
│   ├── passo1_gerar_dataset.py
│   ├── passo2_eda_smote.py
│   ├── passo3_modelos.py
│   ├── passo4_avaliacao.py
│   ├── passo5_analise_erros.py
│   ├── passo5b_sarcasmo.py            (experimento independente)
│   └── passo5c_sarcasmo_ampliado.py   (experimento independente)
├── resultados/                # gerado ao rodar os scripts (gráficos, csv, modelos)
├── relatorio_final_nexus.docx # relatório final (Passo 6)
└── requirements.txt
```

## Como rodar

1. Crie um ambiente virtual (opcional, mas recomendado) e instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

2. Rode os scripts **nessa ordem**, sempre a partir da pasta `scripts/`
   (cada um depende do resultado do anterior, salvo em `resultados/`):

   ```bash
   cd scripts
   python passo1_gerar_dataset.py
   python passo2_eda_smote.py
   python passo3_modelos.py
   python passo4_avaliacao.py
   python passo5_analise_erros.py
   ```

   `passo5b_sarcasmo.py` e `passo5c_sarcasmo_ampliado.py` são
   independentes (geram seus próprios dados na hora) — rode em qualquer
   ordem, sem depender dos passos 1-5.

3. Os gráficos, a tabela comparativa e os modelos treinados (`.joblib`)
   aparecem em `resultados/`.

## Observação sobre reprodutibilidade

`passo1_gerar_dataset.py` usa sementes fixas (`random_state=42` /
`random.seed(42)`), então rodar de novo gera exatamente o mesmo dataset
e, em sequência, exatamente os mesmos números vistos no relatório.

## Resumo dos resultados (conjunto de teste, 600 frases)

| Modelo | Acurácia | Precisão | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.988 | 0.982 | 0.986 | 0.984 |
| Random Forest (extra) | 0.988 | 0.986 | 0.981 | 0.984 |
| Naive Bayes | 0.985 | 0.973 | 0.986 | 0.979 |

Comparação oficial pedida pelo trabalho: **Naive Bayes vs. Logistic
Regression** (Random Forest incluído como resultado complementar).

Detalhes completos, análise de erros, o experimento de detecção de
sarcasmo e as recomendações finais estão em `relatorio_final_nexus.docx`.
