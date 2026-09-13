"""
Passo 4: Testar os algoritmos no conjunto de TESTE (nunca visto no treino,
sem SMOTE, distribuição real) e comparar com as métricas exigidas.
Comparação oficial: Naive Bayes vs Logistic Regression.
Random Forest entra como resultado extra/bônus.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
import joblib

dados = joblib.load("../resultados/dados_passo3.joblib")
vectorizer = dados["vectorizer"]
modelos = dados["modelos"]
X_teste = vectorizer.transform(dados["X_teste_frases"])
y_teste = dados["y_teste"]

resultados = []
matrizes = {}

for nome, modelo in modelos.items():
    X_in = np.clip(X_teste.toarray(), 0, None) if nome == "Naive Bayes" else X_teste
    pred = modelo.predict(X_in)
    resultados.append({
        "Modelo": nome,
        "Acurácia": accuracy_score(y_teste, pred),
        "Precisão": precision_score(y_teste, pred),
        "Recall": recall_score(y_teste, pred),
        "F1-score": f1_score(y_teste, pred),
    })
    matrizes[nome] = confusion_matrix(y_teste, pred)

tabela = pd.DataFrame(resultados).set_index("Modelo").round(3)
print("=" * 70)
print("COMPARAÇÃO DOS MODELOS NO CONJUNTO DE TESTE")
print("=" * 70)
print(tabela)
tabela.to_csv("../resultados/comparacao_modelos.csv")

# matrizes de confusão (as 2 oficiais + a bônus, lado a lado)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, nome in zip(axes, ["Naive Bayes", "Logistic Regression", "Random Forest"]):
    disp = ConfusionMatrixDisplay(matrizes[nome], display_labels=["Sem risco", "Com risco"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(nome, fontsize=10)
fig.suptitle("Matrizes de confusão no conjunto de teste")
fig.tight_layout()
fig.savefig("../resultados/matrizes_confusao.png", dpi=150)

print("\nMatrizes de confusão salvas em matrizes_confusao.png")
print("Tabela comparativa salva em comparacao_modelos.csv")
