# Nexus Churn Risk Classifier

Classificação automática de risco de churn em trechos de transcrições de reuniões de vendas, usando NLP clássico (TF-IDF) e Machine Learning supervisionado.

Projeto acadêmico desenvolvido para a disciplina **Data Science and Statistical Computing** (Sprint 3, Challenge 2026 — FIAP), como parte da evolução do **Nexus**, uma ferramenta de inteligência de CRM desenvolvida em parceria com a TOTVS.

> **Nota sobre os dados:** por não haver, no momento, transcrições reais rotuladas disponíveis, este repositório usa um dataset **sintético** (gerado por código, ver `scripts/passo1_gerar_dataset.py`) que simula frases de reunião de vendas. Os resultados aqui documentados refletem esse dataset sintético — ver a seção [Limitações](#limitações) para o que muda com dados reais.

## O que o projeto faz

1. Gera (ou recebe) frases rotuladas como `risco_churn` (0/1).
2. Faz análise exploratória e balanceia as classes com SMOTE — aplicado somente no conjunto de treino.
3. Treina e compara dois algoritmos de classificação de texto: **Naive Bayes** e **Logistic Regression** (Random Forest incluído como resultado complementar).
4. Avalia com acurácia, precisão, recall, F1-score e matriz de confusão.
5. Analisa os erros (falsos positivos/negativos) e discute suas causas.
6. Gera um relatório final consolidado.
7. Aplica o modelo treinado a uma transcrição real nova (inferência), sem precisar retreinar.

## Resultados (conjunto de teste, 600 frases sintéticas)

| Modelo | Acurácia | Precisão | Recall | F1-score |
|---|---|---|---|---|
| **Logistic Regression** | 0.988 | 0.982 | 0.986 | **0.984** |
| Random Forest (extra) | 0.988 | 0.986 | 0.981 | 0.984 |
| **Naive Bayes** | 0.985 | 0.973 | 0.986 | 0.979 |

Comparação oficial do trabalho: **Naive Bayes vs. Logistic Regression**. Detalhes completos, matrizes de confusão e análise de erros em [`relatorio_final_nexus.docx`](./relatorio_final_nexus.docx).

## Estrutura do repositório

```
.
├── dados/
│   ├── transcricoes_treino.csv        # dados sintéticos de treino (frase, risco_churn)
│   ├── transcricoes_teste.csv         # dados sintéticos de teste
│   └── transcricao_exemplo.txt        # exemplo de transcrição bruta p/ inferência
├── scripts/
│   ├── passo1_gerar_dataset.py        # gera o dataset sintético
│   ├── passo2_eda_smote.py            # análise exploratória + balanceamento (SMOTE)
│   ├── passo3_modelos.py              # treina Naive Bayes, Logistic Regression, Random Forest
│   ├── passo4_avaliacao.py            # métricas e matrizes de confusão no teste
│   ├── passo5_analise_erros.py        # falsos positivos/negativos + palavras mais importantes
│   ├── passo5b_sarcasmo.py            # experimento exploratório: detecção de sarcasmo (dataset pequeno)
│   ├── passo5c_sarcasmo_ampliado.py   # mesmo experimento, dataset maior e estruturado
│   └── passo6_classificar_transcricao_real.py  # aplica o modelo treinado a uma transcrição nova
├── resultados/                        # gerado ao rodar os scripts (gráficos, csv, modelos .joblib)
├── relatorio_final_nexus.docx
├── requirements.txt
└── README.md
```

## Como rodar

```bash
git clone https://github.com/<seu-usuario>/nexus-churn-risk-classifier.git
cd nexus-churn-risk-classifier
pip install -r requirements.txt

cd scripts
python passo1_gerar_dataset.py
python passo2_eda_smote.py
python passo3_modelos.py
python passo4_avaliacao.py
python passo5_analise_erros.py
```

`passo5b_sarcasmo.py` e `passo5c_sarcasmo_ampliado.py` são independentes (geram seus próprios dados na hora).

Pra classificar uma transcrição nova com o modelo já treinado:

```bash
python passo6_classificar_transcricao_real.py
```

(edite a variável `ARQUIVO_TRANSCRICAO` no topo do script pra apontar pro seu arquivo `.txt`).

**Reprodutibilidade:** todos os scripts usam sementes fixas (`random_state=42`), então rodar do zero reproduz exatamente os números documentados aqui.

## Metodologia

- **Vetorização:** TF-IDF (`sklearn.feature_extraction.text.TfidfVectorizer`).
- **Balanceamento:** SMOTE implementado manualmente com `NearestNeighbors` do scikit-learn (sem dependência externa), aplicado **apenas no treino**, elevando a classe minoritária de 36% para 50%.
- **Modelos:** Naive Bayes (`MultinomialNB`), Logistic Regression, Random Forest — todos do scikit-learn.
- **Avaliação:** feita exclusivamente no conjunto de teste, que nunca passou por balanceamento, preservando a distribuição real das classes.

## Limitações

- **Dataset sintético:** boa parte das frases foi gerada por templates, o que torna os padrões lexicais mais consistentes (e a tarefa mais fácil) do que em transcrições reais. As métricas acima **não** devem ser lidas como desempenho esperado em produção.
- **Sem normalização morfológica:** o modelo trata "demora" e "demorado" como palavras diferentes.
- **Cegueira à negação:** modelos bag-of-words (TF-IDF + NB/LR) não captam a inversão de sentido causada por palavras como "não" — confirmado inclusive ao testar com uma transcrição real de exemplo (ver `passo6_classificar_transcricao_real.py`).
- **Detecção de sarcasmo:** testada como experimento à parte (`passo5b`/`passo5c`); abordagens bag-of-words só funcionam quando o sarcasmo segue um padrão lexical artificialmente consistente, não generalizando para sarcasmo real.

## Próximos passos

- Substituir o dataset sintético por transcrições reais rotuladas.
- Aplicar stemming/lematização em português (NLTK RSLPStemmer ou spaCy).
- Tratar negação explicitamente (bigramas/trigramas ou marcação de escopo).
- Avaliar um `VotingClassifier` combinando Naive Bayes e Logistic Regression.
- Para sarcasmo real, investigar modelos com representação contextual (embeddings contextuais / transformers).

## Contexto

Projeto Nexus — parceria FIAP / TOTVS, Challenge 2026, disciplina Data Science and Statistical Computing (Sprint 3).

## Autor

Lucas Zanella Clemente — [GitHub](https://github.com/LucasZanellaClemente)
