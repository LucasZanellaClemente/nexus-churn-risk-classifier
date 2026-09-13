"""
Passo 3: Preparar os algoritmos de classificação.
Usa os dados já vetorizados (TF-IDF) e balanceados (SMOTE) do Passo 2.
Aqui só instanciamos e treinamos os modelos — a avaliação com métricas
fica pro Passo 4.
"""
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

dados = joblib.load("../resultados/dados_passo2.joblib")
vectorizer = dados["vectorizer"]
X_train_bal = dados["X_train_bal"]
y_train_bal = dados["y_train_bal"]

# MultinomialNB não aceita valores negativos; o TF-IDF puro já é >= 0,
# então não há conflito, mas o SMOTE pode gerar interpolações negativas
# perto de zero — corrigimos truncando em 0.
import numpy as np
X_train_bal_nb = np.clip(X_train_bal, a_min=0, a_max=None)

modelos = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Naive Bayes": MultinomialNB(),
}

treinados = {}
for nome, modelo in modelos.items():
    X = X_train_bal_nb if nome == "Naive Bayes" else X_train_bal
    modelo.fit(X, y_train_bal)
    acc_treino = modelo.score(X, y_train_bal)
    print(f"{nome}: treinado com sucesso | acurácia no próprio treino = {acc_treino:.3f}")
    treinados[nome] = modelo

joblib.dump(
    {
        "vectorizer": vectorizer,
        "modelos": treinados,
        "X_teste_frases": dados["X_teste_frases"],
        "y_teste": dados["y_teste"],
    },
    "../resultados/dados_passo3.joblib",
)
print("\nModelos salvos em dados_passo3.joblib, prontos pro Passo 4 (avaliação).")
